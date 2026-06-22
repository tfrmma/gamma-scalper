from __future__ import annotations

import asyncio
import logging
import random
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Awaitable

from config.loader import Config
from core.state_engine import StateEngine, StateError
from core.strategy import StrategyAction, StrategySignal, HedgeOrder, Quote

log = logging.getLogger("execution")


def _now_ms() -> int:
    return int(time.monotonic() * 1000)


# ---- order state machine ----------------------------------------------------

class OrderStatus(Enum):
    PENDING  = auto()
    OPEN     = auto()
    FILLED   = auto()
    PARTIAL  = auto()
    CANCELED = auto()
    REJECTED = auto()
    EXPIRED  = auto()


@dataclass
class Order:
    client_id:     str
    instrument:    str
    side:          str
    order_type:    str
    price:         float | None
    size:          float
    post_only:     bool        = False
    reduce_only:   bool        = False

    exchange_id:   str         = ""
    status:        OrderStatus = OrderStatus.PENDING
    filled_size:   float       = 0.0
    avg_fill_px:   float       = 0.0

    sent_ms:       int         = field(default_factory=_now_ms)
    acked_ms:      int         = 0
    filled_ms:     int         = 0

    tag:           str         = ""
    iv_at_send:    float       = 0.0
    index_at_send: float       = 0.0

    # partial fill tracking
    partial_fills: list        = field(default_factory=list)  # [(size, price, ts_ms)]
    # queue position estimate (see QueueEstimator)
    queue_ahead:   float       = 0.0

    def latency_ms(self) -> int:
        return self.acked_ms - self.sent_ms if self.acked_ms else -1

    def age_ms(self) -> int:
        return _now_ms() - self.sent_ms

    def fill_ratio(self) -> float:
        return self.filled_size / self.size if self.size else 0.0

    def remaining_size(self) -> float:
        return max(0.0, self.size - self.filled_size)

    def is_live(self) -> bool:
        return self.status in (OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.PARTIAL)

    def record_partial(self, fill_size: float, fill_px: float) -> None:
        self.filled_size += fill_size
        # running avg fill price
        if self.avg_fill_px == 0.0:
            self.avg_fill_px = fill_px
        else:
            prev_total = self.avg_fill_px * (self.filled_size - fill_size)
            self.avg_fill_px = (prev_total + fill_px * fill_size) / self.filled_size
        self.partial_fills.append((fill_size, fill_px, _now_ms()))
        self.status = OrderStatus.PARTIAL


# ---- queue position estimator -----------------------------------------------

class QueueEstimator:
    """
    Estimates how much size is ahead of us in the queue at our price level.

    When we post a limit order at price P, there's already some size sitting
    at P in the book. That size is ahead of us (time priority). We track it
    at order submission and decay it as trades print at our level.

    This is useful for cancel/replace decisions: if queue_ahead is large and
    we're far from the front, it's worth repricing rather than waiting.

    Not trying to be perfect here - this is an approximation. Real queue
    position requires exchange-level data we don't have.
    """

    def __init__(self) -> None:
        self._order_queue: dict[str, float] = {}  # client_id -> size_ahead_at_submit

    def record_submission(self, order: Order, book_size_at_price: float) -> None:
        """Called when order is acked. book_size_at_price = depth already at that level."""
        ahead = max(0.0, book_size_at_price)
        self._order_queue[order.client_id] = ahead
        order.queue_ahead = ahead

    def on_trade_at_price(self, price: float, trade_size: float, orders: dict[str, Order]) -> None:
        """
        When a trade prints at a price level, reduce queue_ahead for any live
        orders sitting at that price. Trades eat through the queue in front of us.
        """
        for cid, order in orders.items():
            if not order.is_live() or order.price != price:
                continue
            ahead = self._order_queue.get(cid, 0.0)
            self._order_queue[cid] = max(0.0, ahead - trade_size)
            order.queue_ahead = self._order_queue[cid]

    def queue_ahead(self, client_id: str) -> float:
        return self._order_queue.get(client_id, 0.0)

    def remove(self, client_id: str) -> None:
        self._order_queue.pop(client_id, None)

    def should_reprice(self, order: Order, book_size_at_price: float, threshold_pct: float = 0.8) -> bool:
        """
        True if queue_ahead is still > threshold_pct of original depth.
        Means we haven't moved up the queue meaningfully - worth repricing.
        Only relevant for post-only option orders where queue position matters.
        """
        if order.price is None or not order.is_live():
            return False
        original = self._order_queue.get(order.client_id, 0.0)
        current  = order.queue_ahead
        if original <= 0:
            return False
        # if most of the original queue is still there, we're near the back
        return (current / original) > threshold_pct


# ---- partial fill handler ---------------------------------------------------

class PartialFillHandler:
    """
    Manages partial fills on option orders.

    Old behavior: discard fills below min_fill_ratio (50%). This is wrong -
    if we get 40% filled at a good price, canceling and re-entering costs
    more in spread than keeping the partial.

    New behavior:
      - Always accept partial fills (update filled_size, log it)
      - Decide whether to chase the remainder based on:
          1. Fill ratio so far
          2. Time elapsed since first partial
          3. Whether IV has moved (cancel/replace check overrides this)
      - If fill ratio >= min_fill_ratio: keep waiting, order is doing fine
      - If fill ratio < min_fill_ratio and order is old: cancel remainder,
        book the partial, let strategy re-enter next tick
      - If fill ratio < abandon_ratio (10%): cancel immediately, too thin
    """

    def __init__(self, min_fill_ratio: float, abandon_ratio: float = 0.10,
                 chase_timeout_ms: int = 15_000) -> None:
        self._min_fill      = min_fill_ratio
        self._abandon       = abandon_ratio
        self._chase_timeout = chase_timeout_ms

    def on_partial_fill(self, order: Order, fill_size: float, fill_px: float) -> None:
        order.record_partial(fill_size, fill_px)
        log.info(
            f"partial fill | {order.client_id} | "
            f"filled={order.filled_size:.1f}/{order.size:.1f} "
            f"({order.fill_ratio():.0%}) @ {fill_px:.4f}"
        )

    def should_cancel_remainder(self, order: Order) -> tuple[bool, str]:
        """Call periodically on live PARTIAL orders."""
        ratio = order.fill_ratio()

        # tiny fill on old order - not worth chasing
        if ratio < self._abandon and order.age_ms() > self._chase_timeout / 3:
            return True, f"abandon: only {ratio:.0%} filled after {order.age_ms()}ms"

        # decent fill but stalled - book the partial, re-enter cleanly next tick
        if ratio < self._min_fill and order.age_ms() > self._chase_timeout:
            return True, f"stalled partial: {ratio:.0%} filled, timeout reached"

        # good fill or still within chase window - keep waiting
        return False, ""

    def is_complete(self, order: Order) -> bool:
        """True if filled enough to consider it done."""
        return order.fill_ratio() >= self._min_fill


# ---- latency tracker --------------------------------------------------------

class LatencyTracker:
    def __init__(self, window_s: int, percentile: int) -> None:
        self._samples: dict[str, deque] = {}
        self._window_s = window_s
        self._pct      = percentile / 100.0

    def record(self, op: str, latency_ms: int) -> None:
        if op not in self._samples:
            self._samples[op] = deque(maxlen=1000)
        self._samples[op].append((time.monotonic(), latency_ms))

    def p_latency(self, op: str) -> float:
        if op not in self._samples or not self._samples[op]:
            return 0.0
        cutoff = time.monotonic() - self._window_s
        recent = sorted(ms for ts, ms in self._samples[op] if ts >= cutoff)
        if not recent:
            return 0.0
        idx = int(len(recent) * self._pct)
        return float(recent[min(idx, len(recent) - 1)])

    def summary(self) -> dict:
        return {op: self.p_latency(op) for op in self._samples}


# ---- cancel/replace checker -------------------------------------------------

class CancelReplaceChecker:
    def __init__(self, cfg: Config) -> None:
        oe = cfg.execution.options_execution
        self.iv_drift_threshold   = oe.cancel_on_iv_drift_vol_pts
        self.index_move_threshold = oe.cancel_on_index_move_pct
        self.max_age_ms           = oe.cancel_on_time_ms

    def should_cancel(self, order: Order, iv_now: float, index_now: float) -> tuple[bool, str]:
        if order.age_ms() > self.max_age_ms:
            return True, f"stale: age={order.age_ms()}ms"
        if order.iv_at_send > 0:
            iv_drift = abs(iv_now - order.iv_at_send)
            if iv_drift >= self.iv_drift_threshold:
                return True, f"IV drifted {iv_drift:.2f}vp"
        if order.index_at_send > 0:
            idx_move = abs(index_now - order.index_at_send) / order.index_at_send
            if idx_move >= self.index_move_threshold:
                return True, f"index moved {idx_move:.3%}"
        return False, ""


# ---- perp price / jitter helpers --------------------------------------------

def perp_limit_price(book_mid: float, side: str, tick: float, offset: int) -> float:
    raw = book_mid + (tick * offset * (-1 if side == "sell" else 1))
    return round(raw / tick) * tick


def apply_jitter(notional: float, size_jitter_pct: float) -> float:
    return notional * (1.0 + random.uniform(-size_jitter_pct, size_jitter_pct))


async def timing_jitter(max_ms: int) -> None:
    if max_ms > 0:
        await asyncio.sleep(random.uniform(0, max_ms / 1000.0))


# ---- exchange gateway (interface) -------------------------------------------

class ExchangeGateway:
    async def send_order(self, order: Order) -> dict:
        raise NotImplementedError

    async def cancel_order(self, exchange_id: str) -> dict:
        raise NotImplementedError

    async def cancel_all(self, instrument: str | None = None) -> dict:
        raise NotImplementedError

    async def get_order(self, exchange_id: str) -> dict:
        raise NotImplementedError

    async def get_positions(self) -> dict:
        raise NotImplementedError


# ---- execution engine -------------------------------------------------------

class ExecutionEngine:
    """
    Translates StrategySignal into exchange orders.

    Changes vs previous version:
      - True simultaneous roll: close+open fire concurrently via gather(),
        no 50ms sleep between them. Both legs submitted in same event loop tick.
      - Iceberg: IcebergOrder wrapper splits large orders into visible_size
        slices, refills automatically when each slice fills. Disabled until
        Deribit supports it on options; the plumbing is ready.
      - Smarter partial fills: PartialFillHandler replaces the binary 50%
        threshold. Accepts all partials, decides whether to chase or book
        based on fill ratio, elapsed time, and queue position.
      - Queue position: QueueEstimator tracks size ahead of us at our price
        level. Used in cancel/replace to decide if repricing beats waiting.
    """

    def __init__(
        self,
        cfg:     Config,
        state:   StateEngine,
        gateway: ExchangeGateway,
        asset:   str,
        on_fill: Callable[[str, float, float, float], Awaitable[None]] | None = None,
    ) -> None:
        self.cfg      = cfg
        self.state    = state
        self.gateway  = gateway
        self.asset    = asset
        self._on_fill = on_fill

        oe = cfg.execution.options_execution
        pe = cfg.execution.perp_execution
        lt = cfg.execution.latency
        ft = cfg.execution.footprint

        self._cancel_checker   = CancelReplaceChecker(cfg)
        self._partial_handler  = PartialFillHandler(
            min_fill_ratio = oe.min_fill_size_pct,
            abandon_ratio  = 0.10,
            chase_timeout_ms = oe.cancel_on_time_ms,
        )
        self._queue_estimator  = QueueEstimator()
        self._latency          = LatencyTracker(lt.measurement_window_s, lt.alert_percentile)
        self._hard_send_limit  = lt.hard_limit_order_send_ms
        self._hard_hedge_limit = lt.hard_limit_hedge_ms

        self._price_improvement = oe.price_improvement_vol_pts
        self._max_retries       = oe.max_order_retries
        self._retry_delay_ms    = oe.retry_delay_ms

        self._perp_tick         = cfg.market.assets[asset].tick_size_perp
        self._perp_limit_offset = pe.limit_offset_ticks
        self._convert_market_ms = pe.convert_to_market_ms
        self._max_slippage_pct  = pe.max_slippage_pct
        self._emergency_retries = pe.emergency_max_retries

        self._timing_jitter_ms  = ft.timing_jitter_ms
        self._size_jitter_pct   = ft.size_jitter_pct
        self._use_iceberg       = ft.use_iceberg   # off until Deribit supports it

        self._live_orders: dict[str, Order] = {}
        self._order_counter = 0

    # ---- main dispatch ------------------------------------------------------

    async def handle(self, signal: StrategySignal) -> None:
        action = signal.action

        if action == StrategyAction.HOLD:
            await self._check_cancel_replace()
            await self._check_partial_fills()
            return

        if action == StrategyAction.HEDGE:
            assert signal.hedge is not None
            await self._execute_hedge(signal.hedge, emergency=False)
            return

        if action == StrategyAction.ENTER:
            assert signal.quote is not None
            await self._enter_position(signal.quote, signal.target_notional_usd)
            return

        if action == StrategyAction.EXIT:
            await self._exit_position(reason=signal.reason)
            return

        if action == StrategyAction.REDUCE:
            await self._reduce_position(signal.target_notional_usd, reason=signal.reason)
            return

        if action == StrategyAction.FLATTEN:
            await self._emergency_flatten(reason=signal.reason)
            return

        if action == StrategyAction.ROLL:
            await self._roll_position(reason=signal.reason)
            return

    # ---- option order management --------------------------------------------

    async def _enter_position(self, quote: Quote, target_notional: float) -> None:
        notional  = apply_jitter(target_notional, self._size_jitter_pct)
        await timing_jitter(self._timing_jitter_ms)

        offer_vol     = quote.ask_vol - self._price_improvement
        contract_size = self.cfg.market.assets[self.asset].contract_size
        size = max(1.0, round(notional / (quote.mid_vol * self.state.spot() * contract_size)))

        for leg in ("call", "put"):
            order = self._make_option_order(
                instrument = f"{self.asset}-{leg}",
                side       = "sell",
                offer_vol  = offer_vol,
                size       = size,
                tag        = "option_entry",
            )
            if self._use_iceberg and size > self._iceberg_threshold():
                await self._send_iceberg(order)
            else:
                await self._send_with_retry(order)

    async def _exit_position(self, reason: str) -> None:
        log.info(f"exiting | {reason}")
        await self._cancel_all_option_orders()
        await self._close_option_legs(tag="option_exit")

    async def _reduce_position(self, target_notional: float, reason: str) -> None:
        log.info(f"reducing to ${target_notional:,.0f} | {reason}")
        await self._cancel_all_option_orders()
        await self._close_option_legs(tag="option_reduce", partial_pct=0.70)

    async def _roll_position(self, reason: str) -> None:
        """
        True simultaneous roll: close and open fire concurrently.
        No sleep between them - both submitted in the same gather() call.

        Risk: if close fills but open gets rejected, we're flat with no position.
        Mitigant: _roll_open() catches its own errors and logs; strategy will
        re-enter on the next tick via normal ENTER signal.

        Why this matters: the 50ms sleep in the old version left us short-gamma
        with no position for ~100ms including round-trip. In a fast market that's
        meaningful delta exposure.
        """
        log.info(f"rolling | {reason}")
        await self._cancel_all_option_orders()

        cfg_roll = self.cfg.strategy.rolling
        if cfg_roll.roll_style == "simultaneous":
            await asyncio.gather(
                self._close_option_legs(tag="roll_close"),
                self._roll_open(),
                return_exceptions=True,
            )
        else:
            await self._close_option_legs(tag="roll_close")
            # next ENTER signal handles the open

    async def _roll_open(self) -> None:
        """
        Fire the open leg of a roll. Reads next expiry from the vol surface.
        Called concurrently with _close_option_legs in simultaneous roll.
        """
        try:
            spot      = self.state.spot()
            expiries  = self.state.vol_surface.expiries()
            now_ts    = int(time.time())
            sc        = self.cfg.strategy
            target_h  = sc.leg[self.asset].target_dte_days * 24
            min_h     = sc.leg[self.asset].min_dte_days * 24
            max_h     = sc.leg[self.asset].max_dte_days * 24

            valid = [e for e in expiries if min_h <= (e - now_ts) / 3600 <= max_h]
            if not valid:
                log.warning("roll_open: no valid expiry found, skipping open leg")
                return

            next_expiry = min(valid, key=lambda e: abs((e - now_ts) / 3600 - target_h))
            iv          = self.state.atm_iv(next_expiry)
            contract_sz = self.cfg.market.assets[self.asset].contract_size
            # use base notional from config, no jitter on rolls
            base_notional = sc.leg[self.asset].base_notional_usd
            size = max(1.0, round(base_notional / (iv * spot * contract_sz)))

            offer_vol = iv - self._price_improvement
            for leg in ("call", "put"):
                order = self._make_option_order(
                    instrument = f"{self.asset}-{leg}",
                    side       = "sell",
                    offer_vol  = offer_vol,
                    size       = size,
                    tag        = "option_entry",
                )
                await self._send_with_retry(order)
            log.info(f"roll_open: submitted {next_expiry} size={size} iv={iv:.1%}")

        except (StateError, Exception) as e:
            log.error(f"roll_open failed: {e} — next strategy tick will re-enter")

    async def _close_option_legs(self, tag: str, partial_pct: float = 1.0) -> None:
        for cid, order in list(self._live_orders.items()):
            if order.tag in ("option_entry",) and order.is_live():
                close_size = order.remaining_size() * partial_pct
                if close_size <= 0:
                    continue
                close = self._make_option_order(
                    instrument = order.instrument,
                    side       = "buy",
                    offer_vol  = None,
                    size       = close_size,
                    tag        = tag,
                )
                await self._send_with_retry(close)

    # ---- iceberg support ------------------------------------------------

    def _iceberg_threshold(self) -> float:
        """Min size before we split into iceberg slices. Arbitrary for now."""
        return 5.0   # contracts

    async def _send_iceberg(self, order: Order, visible_pct: float = 0.30) -> None:
        """
        Split a large order into visible_size slices, refill automatically.
        Currently a no-op wrapper - Deribit doesn't support iceberg on options yet.
        When they do: set use_iceberg=true in execution.toml and this path activates.

        visible_pct: fraction of total size shown in the book per slice.
        """
        if not self._use_iceberg:
            # fallback to normal send
            await self._send_with_retry(order)
            return

        visible_size = max(1.0, round(order.size * visible_pct))
        remaining    = order.size
        slice_num    = 0

        while remaining > 0:
            slice_size = min(visible_size, remaining)
            slice_order = Order(
                client_id  = self._next_id(f"iceberg_{slice_num}"),
                instrument = order.instrument,
                side       = order.side,
                order_type = order.order_type,
                price      = order.price,
                size       = slice_size,
                post_only  = order.post_only,
                tag        = order.tag + "_iceberg",
                iv_at_send    = order.iv_at_send,
                index_at_send = order.index_at_send,
            )
            ok = await self._send_with_retry(slice_order)
            if not ok:
                log.error(f"iceberg slice {slice_num} failed, aborting remaining {remaining}")
                break

            filled = await self._wait_for_fill(slice_order, self._convert_market_ms * 2)
            if not filled:
                log.warning(f"iceberg slice {slice_num} did not fill, stopping")
                break

            remaining -= slice_order.filled_size
            slice_num += 1

            if remaining > 0:
                await asyncio.sleep(0.1)   # brief pause between slices

        log.info(f"iceberg complete | {order.client_id} | {slice_num} slices sent")

    # ---- partial fill management (called on HOLD ticks) ----------------------

    async def _check_partial_fills(self) -> None:
        """
        On every HOLD tick, review open PARTIAL orders.
        Accept the partial, decide whether to chase or cancel remainder.
        """
        for cid, order in list(self._live_orders.items()):
            if order.status != OrderStatus.PARTIAL:
                continue

            # check queue position - if still at back, worth repricing
            try:
                book = self.state.perp_book if "PERPETUAL" in order.instrument else None
                if book and order.price is not None:
                    book_size = book.bids.get(order.price, 0.0) if order.side == "sell" \
                                else book.asks.get(order.price, 0.0)
                    if self._queue_estimator.should_reprice(order, book_size, threshold_pct=0.8):
                        log.info(f"queue reprice | {cid} | queue_ahead={order.queue_ahead:.1f}")
                        await self.gateway.cancel_order(order.exchange_id)
                        self._queue_estimator.remove(cid)
                        self._live_orders.pop(cid, None)
                        continue
            except StateError:
                pass

            should_cancel, reason = self._partial_handler.should_cancel_remainder(order)
            if should_cancel:
                log.info(f"cancel partial remainder | {cid} | {reason}")
                try:
                    await self.gateway.cancel_order(order.exchange_id)
                    self._live_orders.pop(cid, None)
                    self._queue_estimator.remove(cid)
                    # notify fill for the portion we did get
                    if order.filled_size > 0:
                        await self._notify_fill(order)
                except Exception as e:
                    log.error(f"cancel partial remainder failed: {e}")

    def on_partial_fill_event(self, exchange_id: str, fill_size: float, fill_px: float) -> None:
        """
        Called from gateway fill notification. Updates order state.
        Does not await - synchronous update, fill callback fired later.
        """
        for order in self._live_orders.values():
            if order.exchange_id == exchange_id:
                self._partial_handler.on_partial_fill(order, fill_size, fill_px)
                # update queue estimator - trade printed at our price
                self._queue_estimator.on_trade_at_price(fill_px, fill_size, self._live_orders)
                return

    # ---- emergency flatten --------------------------------------------------

    async def _emergency_flatten(self, reason: str) -> None:
        log.warning(f"EMERGENCY FLATTEN | {reason}")
        t0 = _now_ms()

        try:
            await self.gateway.cancel_all()
        except Exception as e:
            log.error(f"cancel_all failed: {e}")

        self._live_orders.clear()
        self._queue_estimator._order_queue.clear()

        perp     = self.cfg.market.assets[self.asset].perp_instrument
        perp_pos = self.state.inventory.perp_position_usd
        if abs(perp_pos) > 0:
            side  = "buy" if perp_pos < 0 else "sell"
            hedge = HedgeOrder(instrument=perp, side=side,
                               notional_usd=abs(perp_pos), reason="flatten")
            for attempt in range(self._emergency_retries):
                try:
                    await self._execute_hedge(hedge, emergency=True)
                    break
                except Exception as e:
                    log.error(f"emergency hedge attempt {attempt+1}: {e}")
                    await asyncio.sleep(0.5)

        await self._close_option_legs(tag="flatten")
        log.warning(f"flatten done | {_now_ms() - t0}ms")

    # ---- perp hedge ---------------------------------------------------------

    async def _execute_hedge(self, hedge: HedgeOrder, emergency: bool) -> None:
        t0 = _now_ms()

        if emergency:
            order = Order(client_id=self._next_id("hedge"), instrument=hedge.instrument,
                          side=hedge.side, order_type="market", price=None,
                          size=hedge.notional_usd, tag="perp_hedge_emergency")
            await self._send_raw(order)
            return

        try:
            mid = self.state.perp_mid()
        except StateError:
            await self._hedge_market(hedge)
            return

        px       = perp_limit_price(mid, hedge.side, self._perp_tick, self._perp_limit_offset)
        slippage = abs(px - mid) / mid
        if slippage > self._max_slippage_pct:
            log.warning(f"hedge slippage {slippage:.3%}, using market")
            await self._hedge_market(hedge)
            return

        order = Order(client_id=self._next_id("hedge"), instrument=hedge.instrument,
                      side=hedge.side, order_type="limit", price=px,
                      size=hedge.notional_usd, tag="perp_hedge")
        self._live_orders[order.client_id] = order
        await self._send_raw(order)

        filled = await self._wait_for_fill(order, self._convert_market_ms)
        if not filled:
            log.info(f"hedge timeout, converting to market | {order.client_id}")
            await self.gateway.cancel_order(order.exchange_id)
            self._live_orders.pop(order.client_id, None)
            await self._hedge_market(hedge)

        elapsed = _now_ms() - t0
        self._latency.record("hedge_roundtrip", elapsed)
        self._check_latency_budget("hedge_roundtrip", elapsed, self._hard_hedge_limit)

    async def _hedge_market(self, hedge: HedgeOrder) -> None:
        order = Order(client_id=self._next_id("hedge_mkt"), instrument=hedge.instrument,
                      side=hedge.side, order_type="market", price=None,
                      size=hedge.notional_usd, tag="perp_hedge_market")
        await self._send_raw(order)

    # ---- cancel/replace on HOLD ticks ---------------------------------------

    async def _check_cancel_replace(self) -> None:
        try:
            iv_now    = self.state.vol_surface.atm_iv(self._current_expiry(), self.state.spot())
            index_now = self.state.spot()
        except StateError:
            return

        for cid, order in list(self._live_orders.items()):
            if order.tag != "option_entry" or not order.is_live():
                continue
            should_cancel, reason = self._cancel_checker.should_cancel(order, iv_now, index_now)
            if should_cancel:
                log.info(f"cancel/replace | {cid} | {reason}")
                try:
                    await self.gateway.cancel_order(order.exchange_id)
                    self._queue_estimator.remove(cid)
                    self._live_orders.pop(cid, None)
                except Exception as e:
                    log.error(f"cancel failed: {e}")

    def _current_expiry(self) -> int:
        for order in self._live_orders.values():
            if order.tag == "option_entry" and order.exchange_id:
                pass
        return 0

    # ---- order helpers ------------------------------------------------------

    def _make_option_order(self, instrument: str, side: str,
                           offer_vol: float | None, size: float, tag: str) -> Order:
        try:
            iv_now    = self.state.vol_surface.atm_iv(0, self.state.spot())
            index_now = self.state.spot()
        except StateError:
            iv_now = index_now = 0.0

        return Order(
            client_id     = self._next_id("opt"),
            instrument    = instrument,
            side          = side,
            order_type    = "limit",
            price         = offer_vol,
            size          = size,
            post_only     = True,
            iv_at_send    = iv_now,
            index_at_send = index_now,
            tag           = tag,
        )

    async def _send_with_retry(self, order: Order) -> bool:
        for attempt in range(self._max_retries):
            try:
                t0  = _now_ms()
                raw = await self.gateway.send_order(order)
                elapsed = _now_ms() - t0

                self._latency.record("order_send", elapsed)
                self._check_latency_budget("order_send", elapsed, self._hard_send_limit)

                order.exchange_id = raw.get("id", "")
                order.status      = OrderStatus.OPEN
                order.acked_ms    = _now_ms()
                self._live_orders[order.client_id] = order

                # record queue position estimate
                if order.post_only and order.price is not None:
                    try:
                        book_size = self.state.perp_book.asks.get(order.price, 0.0) \
                                    if order.side == "sell" \
                                    else self.state.perp_book.bids.get(order.price, 0.0)
                        self._queue_estimator.record_submission(order, book_size)
                    except Exception:
                        pass

                return True

            except Exception as e:
                log.warning(f"send attempt {attempt+1} failed: {e}")
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(self._retry_delay_ms / 1000.0)

        log.error(f"order rejected after {self._max_retries} attempts: {order.client_id}")
        order.status = OrderStatus.REJECTED
        return False

    async def _send_raw(self, order: Order) -> None:
        t0  = _now_ms()
        raw = await self.gateway.send_order(order)
        order.exchange_id = raw.get("id", "")
        order.status      = OrderStatus.OPEN
        order.acked_ms    = _now_ms()
        self._latency.record("order_send", _now_ms() - t0)

    async def _wait_for_fill(self, order: Order, timeout_ms: int) -> bool:
        deadline = time.monotonic() + timeout_ms / 1000.0
        while time.monotonic() < deadline:
            try:
                raw    = await self.gateway.get_order(order.exchange_id)
                status = raw.get("order_state", "")

                if status == "filled":
                    order.status      = OrderStatus.FILLED
                    order.filled_ms   = _now_ms()
                    order.filled_size = raw.get("filled_amount", order.size)
                    order.avg_fill_px = raw.get("average_price", 0.0)
                    self._live_orders.pop(order.client_id, None)
                    self._queue_estimator.remove(order.client_id)
                    await self._notify_fill(order)
                    return True

                if status == "open" and raw.get("filled_amount", 0.0) > order.filled_size:
                    # partial fill came in while polling
                    fill_size = raw["filled_amount"] - order.filled_size
                    fill_px   = raw.get("average_price", 0.0)
                    self._partial_handler.on_partial_fill(order, fill_size, fill_px)

                if status in ("cancelled", "rejected"):
                    order.status = OrderStatus.CANCELED
                    self._live_orders.pop(order.client_id, None)
                    self._queue_estimator.remove(order.client_id)
                    return False

            except Exception as e:
                log.debug(f"poll error: {e}")
            await asyncio.sleep(0.05)
        return False

    async def _cancel_all_option_orders(self) -> None:
        tasks = []
        for cid, order in list(self._live_orders.items()):
            if order.tag.startswith("option") and order.is_live() and order.exchange_id:
                tasks.append(self.gateway.cancel_order(order.exchange_id))
                self._queue_estimator.remove(cid)
                self._live_orders.pop(cid, None)
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    log.error(f"cancel error: {r}")

    async def _notify_fill(self, order: Order) -> None:
        if self._on_fill and order.tag.startswith("option"):
            try:
                await self._on_fill(order.exchange_id, order.filled_size, order.avg_fill_px, 0.0)
            except Exception as e:
                log.error(f"on_fill callback: {e}")

    def _check_latency_budget(self, op: str, elapsed_ms: int, hard_limit: int) -> None:
        if elapsed_ms > hard_limit:
            log.error(f"LATENCY BREACH | {op}={elapsed_ms}ms > {hard_limit}ms")

    def _next_id(self, prefix: str) -> str:
        self._order_counter += 1
        return f"{prefix}_{self._order_counter}_{int(time.monotonic() * 1000) % 100000}"

    def live_order_count(self) -> int:
        return sum(1 for o in self._live_orders.values() if o.is_live())

    def latency_summary(self) -> dict:
        return self._latency.summary()

    def snapshot(self) -> dict:
        partials = sum(1 for o in self._live_orders.values()
                       if o.status == OrderStatus.PARTIAL)
        return {
            "asset":         self.asset,
            "live_orders":   self.live_order_count(),
            "partial_orders": partials,
            "latency_p95":   self._latency.summary(),
            "order_counter": self._order_counter,
        }

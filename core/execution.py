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
    PENDING  = auto()   # sent, waiting ack
    OPEN     = auto()   # acked, live on book
    FILLED   = auto()   # done
    PARTIAL  = auto()   # partial fill, still open
    CANCELED = auto()
    REJECTED = auto()
    EXPIRED  = auto()


@dataclass
class Order:
    client_id:    str
    instrument:   str
    side:         str        # "buy" | "sell"
    order_type:   str        # "limit" | "market"
    price:        float | None
    size:         float
    post_only:    bool       = False
    reduce_only:  bool       = False

    # set after exchange ack
    exchange_id:  str        = ""
    status:       OrderStatus = OrderStatus.PENDING
    filled_size:  float      = 0.0
    avg_fill_px:  float      = 0.0

    sent_ms:      int        = field(default_factory=_now_ms)
    acked_ms:     int        = 0
    filled_ms:    int        = 0

    # what this order is for - helps with cancel/replace decisions
    tag:          str        = ""   # "option_entry" | "perp_hedge" | "flatten" | "roll"
    iv_at_send:   float      = 0.0  # for cancel-on-drift check
    index_at_send: float     = 0.0

    def latency_ms(self) -> int:
        if self.acked_ms == 0:
            return -1
        return self.acked_ms - self.sent_ms

    def age_ms(self) -> int:
        return _now_ms() - self.sent_ms

    def fill_ratio(self) -> float:
        if self.size == 0:
            return 0.0
        return self.filled_size / self.size

    def is_live(self) -> bool:
        return self.status in (OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.PARTIAL)


# ---- latency tracker --------------------------------------------------------

class LatencyTracker:
    """
    Rolling P95 latency per operation type.
    Not doing anything fancy - just a deque with a percentile query.
    If you want Prometheus histograms, add them on top of this.
    """

    def __init__(self, window_s: int, percentile: int) -> None:
        maxlen = window_s * 10   # assume ~10 events/s worst case
        self._samples: dict[str, deque] = {}
        self._window_s  = window_s
        self._pct       = percentile / 100.0

    def record(self, op: str, latency_ms: int) -> None:
        if op not in self._samples:
            self._samples[op] = deque(maxlen=1000)
        self._samples[op].append((time.monotonic(), latency_ms))

    def p_latency(self, op: str) -> float:
        if op not in self._samples or not self._samples[op]:
            return 0.0
        cutoff = time.monotonic() - self._window_s
        recent = [ms for ts, ms in self._samples[op] if ts >= cutoff]
        if not recent:
            return 0.0
        recent.sort()
        idx = int(len(recent) * self._pct)
        return float(recent[min(idx, len(recent) - 1)])

    def summary(self) -> dict:
        return {op: self.p_latency(op) for op in self._samples}


# ---- cancel/replace logic ---------------------------------------------------

class CancelReplaceChecker:
    """
    Decides whether a live option order should be canceled and requoted.
    Three triggers: IV drift, index move, time.
    """

    def __init__(self, cfg: Config) -> None:
        oe = cfg.execution.options_execution
        self.iv_drift_threshold    = oe.cancel_on_iv_drift_vol_pts
        self.index_move_threshold  = oe.cancel_on_index_move_pct
        self.max_age_ms            = oe.cancel_on_time_ms

    def should_cancel(
        self,
        order:     Order,
        iv_now:    float,
        index_now: float,
    ) -> tuple[bool, str]:
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


# ---- perp price calculator --------------------------------------------------

def perp_limit_price(
    book_mid: float,
    side:     str,
    tick:     float,
    offset:   int,
) -> float:
    """
    Aggressive limit: inside the spread by `offset` ticks.
    Rounds to tick size. If it crosses mid, that's intentional - we want a fill.
    """
    raw = book_mid + (tick * offset * (-1 if side == "sell" else 1))
    return round(raw / tick) * tick


# ---- jitter -----------------------------------------------------------------

def apply_jitter(notional: float, size_jitter_pct: float) -> float:
    """±jitter% on size. Enough to not be a pattern, not enough to matter."""
    jitter = 1.0 + random.uniform(-size_jitter_pct, size_jitter_pct)
    return notional * jitter


async def timing_jitter(max_ms: int) -> None:
    """Random delay before sending. Not a lot, just enough."""
    if max_ms > 0:
        await asyncio.sleep(random.uniform(0, max_ms / 1000.0))


# ---- exchange gateway (interface) -------------------------------------------

class ExchangeGateway:
    """
    Abstract interface to the exchange. Swap this out for testnet/live/mock.
    All methods return dicts matching Deribit API response shape.
    Concrete implementation lives in infra/deribit_gateway.py - not here.
    """

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
    Translates StrategySignal into exchange orders. Manages order lifecycle.

    Responsibilities:
      - Send, cancel, replace option orders
      - Execute perp hedges (aggressive limit, convert to market)
      - Track latency, alert on budget breaches
      - Emergency flatten path (synchronous-ish, market orders, no mercy)

    Does NOT make trading decisions. If you find strategy logic here, move it.
    Does NOT hold market state. If you find book reads here, they come from StateEngine.
    """

    def __init__(
        self,
        cfg:     Config,
        state:   StateEngine,
        gateway: ExchangeGateway,
        asset:   str,
        on_fill: Callable[[str, float, float, float], Awaitable[None]] | None = None,
    ) -> None:
        self.cfg     = cfg
        self.state   = state
        self.gateway = gateway
        self.asset   = asset
        self._on_fill = on_fill   # callback to strategy.on_fill + state.on_option_fill

        oe = cfg.execution.options_execution
        pe = cfg.execution.perp_execution
        lt = cfg.execution.latency
        ft = cfg.execution.footprint

        self._cancel_checker  = CancelReplaceChecker(cfg)
        self._latency         = LatencyTracker(lt.measurement_window_s, lt.alert_percentile)
        self._hard_send_limit = lt.hard_limit_order_send_ms
        self._hard_hedge_limit= lt.hard_limit_hedge_ms

        self._price_improvement = oe.price_improvement_vol_pts
        self._min_fill_ratio    = oe.min_fill_size_pct
        self._max_retries       = oe.max_order_retries
        self._retry_delay_ms    = oe.retry_delay_ms

        self._perp_tick         = cfg.market.assets[asset].tick_size_perp
        self._perp_limit_offset = pe.limit_offset_ticks
        self._convert_market_ms = pe.convert_to_market_ms
        self._max_slippage_pct  = pe.max_slippage_pct
        self._emergency_retries = pe.emergency_max_retries

        self._timing_jitter_ms  = ft.timing_jitter_ms
        self._size_jitter_pct   = ft.size_jitter_pct

        # live orders: client_id -> Order
        self._live_orders: dict[str, Order] = {}
        self._order_counter = 0

    # ---- main dispatch ------------------------------------------------------

    async def handle(self, signal: StrategySignal) -> None:
        """Route a strategy signal to the right execution path."""
        action = signal.action

        if action == StrategyAction.HOLD:
            await self._check_cancel_replace()
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
            # emergency - don't await jitter, don't care about slippage
            await self._emergency_flatten(reason=signal.reason)
            return

        if action == StrategyAction.ROLL:
            await self._roll_position(reason=signal.reason)
            return

    # ---- option order management --------------------------------------------

    async def _enter_position(self, quote: Quote, target_notional: float) -> None:
        """Send a short straddle. Two legs: sell call + sell put at the same strike."""
        notional = apply_jitter(target_notional, self._size_jitter_pct)
        await timing_jitter(self._timing_jitter_ms)

        # sell at ask_vol - price_improvement (we're the seller, price towards buyer)
        # i.e. slightly below our theoretical ask to get filled
        offer_vol = quote.ask_vol - self._price_improvement

        # vol -> dollar price conversion happens in the gateway (needs spot, DTE, etc.)
        # we pass vol and let the gateway do Black-76. clean separation.
        contract_size = self.cfg.market.assets[self.asset].contract_size
        size = notional / (quote.mid_vol * self.state.spot() * contract_size)
        size = max(1.0, round(size))

        perp_instrument = self.cfg.market.assets[self.asset].perp_instrument

        for leg in ("call", "put"):
            order = self._make_option_order(
                instrument = f"{self.asset}-{leg}",
                side       = "sell",
                offer_vol  = offer_vol,
                size       = size,
                tag        = "option_entry",
            )
            await self._send_with_retry(order)

    async def _exit_position(self, reason: str) -> None:
        log.info(f"exiting position | {reason}")
        await self._cancel_all_option_orders()
        # buy back - execution layer sends market orders on exit
        # strategy already decided to exit, price doesn't matter much here
        await self._close_option_legs(order_type="limit", tag="option_exit")

    async def _reduce_position(self, target_notional: float, reason: str) -> None:
        log.info(f"reducing to ${target_notional:,.0f} | {reason}")
        # cancel outstanding quotes first, then partial close
        await self._cancel_all_option_orders()
        # reduction sizing: current_notional - target = amount to buy back
        # caller (strategy) has already computed target, we just execute it
        await self._close_option_legs(
            order_type = "limit",
            tag        = "option_reduce",
            partial_pct= 0.70,   # close 70% - rest stays on, good enough
        )

    async def _roll_position(self, reason: str) -> None:
        """Close current expiry, reopen on next target expiry. Best effort simultaneous."""
        log.info(f"rolling position | {reason}")
        await self._cancel_all_option_orders()
        # close existing legs then enter will be triggered by next strategy tick
        # roll_style = "simultaneous" means we fire close+open in the same event loop iteration
        # leg_by_leg would be sequential, less risky but leaves delta exposure between
        cfg_roll = self.cfg.strategy.rolling
        if cfg_roll.roll_style == "simultaneous":
            close_task = asyncio.create_task(
                self._close_option_legs(order_type="limit", tag="roll_close")
            )
            # give close a head start then kick off the open
            await asyncio.sleep(0.05)
            open_task = asyncio.create_task(self._roll_open_placeholder())
            await asyncio.gather(close_task, open_task)
        else:
            await self._close_option_legs(order_type="limit", tag="roll_close")
            # next strategy tick will see no position + positive signal = ENTER

    async def _roll_open_placeholder(self) -> None:
        # actual roll open is handled by the next strategy ENTER signal
        # this task is a hook for simultaneous roll if needed in future
        # TODO: if we want true simultaneous, need to prefetch next expiry quote here
        pass

    async def _close_option_legs(
        self,
        order_type:  str,
        tag:         str,
        partial_pct: float = 1.0,
    ) -> None:
        """Buy back open short option legs."""
        # in real impl, we'd query live positions from state/gateway
        # for now, send buy orders for each open order we have on the books
        for cid, order in list(self._live_orders.items()):
            if order.tag in ("option_entry",) and order.is_live():
                close_size = order.size * partial_pct
                close = self._make_option_order(
                    instrument = order.instrument,
                    side       = "buy",
                    offer_vol  = None,    # market close - gateway picks price
                    size       = close_size,
                    tag        = tag,
                )
                await self._send_with_retry(close)

    async def _emergency_flatten(self, reason: str) -> None:
        """
        Get flat by any means necessary.
        Market orders, retries, cancel everything first.
        Don't call this unless risk engine said so.
        """
        log.warning(f"EMERGENCY FLATTEN | {reason}")
        t0 = _now_ms()

        # nuke all live orders first - don't want fills while we're trying to flatten
        try:
            await self.gateway.cancel_all()
        except Exception as e:
            log.error(f"cancel_all failed during flatten: {e}")

        self._live_orders.clear()

        perp = self.cfg.market.assets[self.asset].perp_instrument

        # close perp hedge - this is the fast path, options can wait a tick
        perp_pos = self.state.inventory.perp_position_usd
        if abs(perp_pos) > 0:
            side = "buy" if perp_pos < 0 else "sell"
            hedge = HedgeOrder(
                instrument   = perp,
                side         = side,
                notional_usd = abs(perp_pos),
                reason       = "emergency flatten",
            )
            for attempt in range(self._emergency_retries):
                try:
                    await self._execute_hedge(hedge, emergency=True)
                    break
                except Exception as e:
                    log.error(f"emergency hedge attempt {attempt+1} failed: {e}")
                    await asyncio.sleep(0.5)

        # close option legs - market orders, accept any price
        await self._close_option_legs(order_type="market", tag="flatten")

        elapsed = _now_ms() - t0
        log.warning(f"flatten complete | elapsed={elapsed}ms")

    # ---- perp hedge execution -----------------------------------------------

    async def _execute_hedge(self, hedge: HedgeOrder, emergency: bool) -> None:
        t0 = _now_ms()

        if emergency:
            order = Order(
                client_id  = self._next_id("hedge"),
                instrument = hedge.instrument,
                side       = hedge.side,
                order_type = "market",
                price      = None,
                size       = hedge.notional_usd,
                tag        = "perp_hedge_emergency",
            )
            await self._send_raw(order)
            return

        # aggressive limit: price inside spread, convert to market on timeout
        try:
            mid = self.state.perp_mid()
        except StateError:
            log.warning("perp book empty during hedge, falling back to market")
            await self._hedge_market(hedge)
            return

        px = perp_limit_price(mid, hedge.side, self._perp_tick, self._perp_limit_offset)

        # slippage check before sending
        slippage = abs(px - mid) / mid
        if slippage > self._max_slippage_pct:
            log.warning(f"hedge slippage {slippage:.3%} > limit {self._max_slippage_pct:.3%}, using market")
            await self._hedge_market(hedge)
            return

        order = Order(
            client_id  = self._next_id("hedge"),
            instrument = hedge.instrument,
            side       = hedge.side,
            order_type = "limit",
            price      = px,
            size       = hedge.notional_usd,
            post_only  = False,   # hedge needs to fill, not sit on book
            tag        = "perp_hedge",
        )

        self._live_orders[order.client_id] = order
        await self._send_raw(order)

        # wait for fill, convert to market if timeout
        filled = await self._wait_for_fill(order, self._convert_market_ms)
        if not filled:
            log.info(f"hedge limit timeout, converting to market | {order.client_id}")
            await self.gateway.cancel_order(order.exchange_id)
            self._live_orders.pop(order.client_id, None)
            await self._hedge_market(hedge)

        elapsed = _now_ms() - t0
        self._latency.record("hedge_roundtrip", elapsed)
        self._check_latency_budget("hedge_roundtrip", elapsed, self._hard_hedge_limit)

    async def _hedge_market(self, hedge: HedgeOrder) -> None:
        order = Order(
            client_id  = self._next_id("hedge_mkt"),
            instrument = hedge.instrument,
            side       = hedge.side,
            order_type = "market",
            price      = None,
            size       = hedge.notional_usd,
            tag        = "perp_hedge_market",
        )
        await self._send_raw(order)

    # ---- cancel/replace on HOLD ticks ---------------------------------------

    async def _check_cancel_replace(self) -> None:
        """Called on every HOLD tick. Cancel stale quotes and requote."""
        try:
            iv_now    = self.state.vol_surface.atm_iv(
                self._current_expiry(), self.state.spot()
            )
            index_now = self.state.spot()
        except StateError:
            return   # state not ready, skip

        for cid, order in list(self._live_orders.items()):
            if order.tag != "option_entry" or not order.is_live():
                continue
            should_cancel, reason = self._cancel_checker.should_cancel(order, iv_now, index_now)
            if should_cancel:
                log.info(f"cancel/replace | {order.client_id} | {reason}")
                try:
                    await self.gateway.cancel_order(order.exchange_id)
                    self._live_orders.pop(cid, None)
                    # requote happens on the next strategy tick - keeps logic clean
                except Exception as e:
                    log.error(f"cancel failed: {e}")

    def _current_expiry(self) -> int:
        # try to get from live option orders
        for order in self._live_orders.values():
            if order.tag == "option_entry" and order.exchange_id:
                # parse from instrument name if needed - placeholder for now
                pass
        return 0   # if we can't find it, StateError will propagate naturally

    # ---- order helpers ------------------------------------------------------

    def _make_option_order(
        self,
        instrument: str,
        side:       str,
        offer_vol:  float | None,
        size:       float,
        tag:        str,
    ) -> Order:
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
            price         = offer_vol,   # gateway converts vol -> price via B76
            size          = size,
            post_only     = True,
            iv_at_send    = iv_now,
            index_at_send = index_now,
            tag           = tag,
        )

    async def _send_with_retry(self, order: Order) -> bool:
        """Send an order, retry on rejection up to max_retries."""
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
                return True

            except Exception as e:
                log.warning(f"order send attempt {attempt+1} failed: {e}")
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(self._retry_delay_ms / 1000.0)

        log.error(f"order failed after {self._max_retries} attempts: {order.client_id}")
        order.status = OrderStatus.REJECTED
        return False

    async def _send_raw(self, order: Order) -> None:
        """Fire and forget. Used for hedges and emergency paths where retry is caller's job."""
        t0 = _now_ms()
        raw = await self.gateway.send_order(order)
        order.exchange_id = raw.get("id", "")
        order.status      = OrderStatus.OPEN
        order.acked_ms    = _now_ms()
        self._latency.record("order_send", _now_ms() - t0)

    async def _wait_for_fill(self, order: Order, timeout_ms: int) -> bool:
        """Poll for fill status. Crude but effective for a 500ms window."""
        deadline = time.monotonic() + timeout_ms / 1000.0
        while time.monotonic() < deadline:
            try:
                raw = await self.gateway.get_order(order.exchange_id)
                status = raw.get("order_state", "")
                if status == "filled":
                    order.status     = OrderStatus.FILLED
                    order.filled_ms  = _now_ms()
                    order.filled_size= raw.get("filled_amount", order.size)
                    order.avg_fill_px= raw.get("average_price", 0.0)
                    self._live_orders.pop(order.client_id, None)
                    await self._notify_fill(order)
                    return True
                if status in ("cancelled", "rejected"):
                    order.status = OrderStatus.CANCELED
                    self._live_orders.pop(order.client_id, None)
                    return False
            except Exception as e:
                log.debug(f"poll error (non-fatal): {e}")
            await asyncio.sleep(0.05)   # 50ms poll interval
        return False

    async def _cancel_all_option_orders(self) -> None:
        tasks = []
        for cid, order in list(self._live_orders.items()):
            if order.tag.startswith("option") and order.is_live() and order.exchange_id:
                tasks.append(self.gateway.cancel_order(order.exchange_id))
                self._live_orders.pop(cid, None)
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    log.error(f"cancel error: {r}")

    async def _notify_fill(self, order: Order) -> None:
        if self._on_fill and order.tag.startswith("option"):
            try:
                # fill size, avg price, and a placeholder for greeks
                # real impl gets greeks from the fill notification, not from here
                await self._on_fill(order.exchange_id, order.filled_size, order.avg_fill_px, 0.0)
            except Exception as e:
                log.error(f"on_fill callback error: {e}")

    # ---- latency monitoring -------------------------------------------------

    def _check_latency_budget(self, op: str, elapsed_ms: int, hard_limit: int) -> None:
        if elapsed_ms > hard_limit:
            # risk engine will pick this up from the latency tracker
            log.error(f"LATENCY BREACH | {op}={elapsed_ms}ms > limit={hard_limit}ms")

    # ---- misc ---------------------------------------------------------------

    def _next_id(self, prefix: str) -> str:
        self._order_counter += 1
        return f"{prefix}_{self._order_counter}_{int(time.monotonic() * 1000) % 100000}"

    def live_order_count(self) -> int:
        return sum(1 for o in self._live_orders.values() if o.is_live())

    def latency_summary(self) -> dict:
        return self._latency.summary()

    def snapshot(self) -> dict:
        return {
            "asset":         self.asset,
            "live_orders":   self.live_order_count(),
            "latency_p95":   self._latency.summary(),
            "order_counter": self._order_counter,
        }

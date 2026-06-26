from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Awaitable

from config.loader import Config
from core.state_engine import StateEngine, StateError
from core.execution import ExecutionEngine

log = logging.getLogger("risk")


# ---- enums ------------------------------------------------------------------

class RiskLevel(Enum):
    OK     = auto()
    WARN   = auto()
    REDUCE = auto()
    HALT   = auto()


class KillReason(Enum):
    RV_SPIKE          = auto()
    FUNDING_NEGATIVE  = auto()
    DRAWDOWN_INTRADAY = auto()
    DRAWDOWN_24H      = auto()
    DRAWDOWN_7D       = auto()
    LOSS_VELOCITY     = auto()
    MARGIN_BREACH     = auto()
    API_LATENCY       = auto()
    API_ERRORS        = auto()
    WS_DISCONNECT     = auto()
    BOOK_STALE        = auto()
    INDEX_DIVERGENCE  = auto()
    POSITION_LIMIT    = auto()
    MANUAL            = auto()


# ---- alerting ---------------------------------------------------------------

class Alerter:
    """
    Sends halt/warn notifications. Reads credentials from env at runtime,
    not from config - don't commit tokens.

    Supported backends:
      TELEGRAM: set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID
      SLACK:    set SLACK_WEBHOOK_URL
      Both can be active simultaneously.

    Falls back to log-only if env vars aren't set, so it won't blow up in
    testnet or CI where you don't have creds.
    """

    def __init__(self) -> None:
        self._tg_token   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self._tg_chat    = os.environ.get("TELEGRAM_CHAT_ID", "")
        self._slack_url  = os.environ.get("SLACK_WEBHOOK_URL", "")
        self._enabled    = bool(self._tg_token or self._slack_url)

        if not self._enabled:
            log.info("alerter: no credentials found, log-only mode")

    async def halt(self, asset: str, reason: KillReason, detail: str) -> None:
        msg = f"KILL SWITCH | {asset} | {reason.name} | {detail}"
        log.error(msg)
        await self._send(f"[HALT] {msg}")

    async def warn(self, asset: str, msg: str) -> None:
        log.warning(f"RISK WARN | {asset} | {msg}")
        await self._send(f"[WARN] {asset} | {msg}")

    async def _send(self, text: str) -> None:
        if not self._enabled:
            return
        tasks = []
        if self._tg_token and self._tg_chat:
            tasks.append(self._send_telegram(text))
        if self._slack_url:
            tasks.append(self._send_slack(text))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                log.error(f"alert send failed: {r}")

    async def _send_telegram(self, text: str) -> None:
        try:
            import aiohttp
        except ImportError:
            log.warning("aiohttp not installed, can't send Telegram alert")
            return
        url = f"https://api.telegram.org/bot{self._tg_token}/sendMessage"
        async with aiohttp.ClientSession() as s:
            await s.post(url, json={"chat_id": self._tg_chat, "text": text}, timeout=5)

    async def _send_slack(self, text: str) -> None:
        try:
            import aiohttp
        except ImportError:
            return
        async with aiohttp.ClientSession() as s:
            await s.post(self._slack_url, json={"text": text}, timeout=5)


# ---- per-leg PnL attribution -----------------------------------------------

@dataclass
class LegPnl:
    """PnL for a single option leg (call or put)."""
    instrument:  str
    gamma_pnl:   float = 0.0
    theta_pnl:   float = 0.0
    vega_pnl:    float = 0.0
    realized_pnl: float = 0.0   # from fills: fill_price - entry_price
    net:          float = 0.0

    def update(self, gamma: float, theta: float, vega: float,
               dS: float, dv: float, dt_h: float) -> None:
        self.gamma_pnl = 0.5 * gamma * (dS ** 2)
        self.theta_pnl = theta * dt_h
        self.vega_pnl  = vega * dv
        self.net       = self.gamma_pnl + self.theta_pnl + self.vega_pnl + self.realized_pnl

    def record_fill(self, fill_price: float, entry_price: float, size: float, side: str) -> None:
        # for short gamma: sold at entry_price, buying back at fill_price
        # profit = (entry - fill) * size for sells
        mult = 1.0 if side == "sell" else -1.0
        self.realized_pnl += mult * (entry_price - fill_price) * size

    def as_dict(self) -> dict:
        return {
            "instrument":   self.instrument,
            "gamma_pnl":    round(self.gamma_pnl, 4),
            "theta_pnl":    round(self.theta_pnl, 4),
            "vega_pnl":     round(self.vega_pnl, 4),
            "realized_pnl": round(self.realized_pnl, 4),
            "net":          round(self.net, 4),
        }


@dataclass
class PnlComponents:
    """Aggregate PnL across all legs + perp hedge."""
    gamma_pnl:  float = 0.0
    theta_pnl:  float = 0.0
    vega_pnl:   float = 0.0
    funding:    float = 0.0
    tx_cost:    float = 0.0
    realized:   float = 0.0   # sum of realized fill PnL across legs
    legs:       dict  = field(default_factory=dict)  # instrument -> LegPnl

    @property
    def total(self) -> float:
        return self.gamma_pnl + self.theta_pnl + self.vega_pnl + self.funding + self.realized - self.tx_cost

    def dominant_component(self) -> tuple[str, float]:
        components = {
            "gamma":    abs(self.gamma_pnl),
            "theta":    abs(self.theta_pnl),
            "vega":     abs(self.vega_pnl),
            "funding":  abs(self.funding),
            "realized": abs(self.realized),
            "tx_cost":  abs(self.tx_cost),
        }
        name      = max(components, key=components.get)
        total_abs = sum(components.values())
        ratio     = components[name] / total_abs if total_abs > 0 else 0.0
        return name, ratio

    def as_dict(self) -> dict:
        return {
            "gamma_pnl":  self.gamma_pnl,
            "theta_pnl":  self.theta_pnl,
            "vega_pnl":   self.vega_pnl,
            "funding":    self.funding,
            "realized":   self.realized,
            "tx_cost":    self.tx_cost,
            "total":      self.total,
            "legs":       {k: v.as_dict() for k, v in self.legs.items()},
        }


class PnlAttributor:
    """
    Per-leg PnL attribution with real fill PnL.

    Greek PnL (gamma/theta/vega) is approximated from bumps - same as before.
    Realized PnL is computed from actual fills: entry_price vs exit_price.
    Per-leg breakdown tells you which leg is leaking (usually puts in a vol spike).

    record_fill() is called by the risk engine when a fill notification arrives
    with the actual fill price and the stored entry price for that instrument.
    """

    def __init__(self, recompute_interval_s: int, dominance_threshold: float) -> None:
        self._interval   = recompute_interval_s
        self._dominance  = dominance_threshold
        self._last_run   = 0.0
        self._last_spot  = 0.0
        self._last_iv    = 0.0
        self._components = PnlComponents()
        self._history: deque[tuple[float, float]] = deque(maxlen=1440)  # (ts, total_pnl)

        # entry prices per instrument, set when we open a leg
        self._entry_prices: dict[str, float] = {}

    def record_entry(self, instrument: str, price: float) -> None:
        self._entry_prices[instrument] = price
        if instrument not in self._components.legs:
            self._components.legs[instrument] = LegPnl(instrument=instrument)

    def record_fill(self, instrument: str, fill_price: float, size: float, side: str) -> None:
        """Real fill PnL: called on every fill notification from the gateway."""
        entry = self._entry_prices.get(instrument, fill_price)
        leg   = self._components.legs.setdefault(instrument, LegPnl(instrument=instrument))
        leg.record_fill(fill_price, entry, size, side)
        self._components.realized = sum(l.realized_pnl for l in self._components.legs.values())
        log.debug(f"fill PnL | {instrument} | entry={entry:.4f} fill={fill_price:.4f} "
                  f"size={size} realized={leg.realized_pnl:.2f}")

    def update(
        self,
        spot:       float,
        iv:         float,
        gamma:      float,
        theta:      float,
        vega:       float,
        funding_8h: float,
        notional:   float,
        fees_paid:  float,
    ) -> PnlComponents:
        now = time.monotonic()
        if now - self._last_run < self._interval:
            return self._components

        dt_h = (now - self._last_run) / 3600.0 if self._last_run > 0 else 0.0

        if self._last_spot > 0 and dt_h > 0:
            dS = spot - self._last_spot
            dv = iv   - self._last_iv

            self._components.gamma_pnl = 0.5 * gamma * (dS ** 2)
            self._components.theta_pnl = theta * dt_h
            self._components.vega_pnl  = vega * dv
            self._components.funding   = funding_8h * notional * (dt_h / 8.0)
            self._components.tx_cost   = fees_paid

            # push greeks down to legs proportionally (equal split across legs for now)
            n_legs = len(self._components.legs)
            if n_legs > 0:
                leg_gamma = gamma / n_legs
                leg_theta = theta / n_legs
                leg_vega  = vega  / n_legs
                for leg in self._components.legs.values():
                    leg.update(leg_gamma, leg_theta, leg_vega, dS, dv, dt_h)

        self._last_spot = spot
        self._last_iv   = iv
        self._last_run  = now
        self._history.append((now, self._components.total))

        name, ratio = self._components.dominant_component()
        if ratio > self._dominance:
            log.warning(f"PnL dominated by {name} ({ratio:.0%}) - check model")

        return self._components

    @property
    def latest(self) -> PnlComponents:
        return self._components


# ---- real PnL from fills ----------------------------------------------------

class FillPnlAccumulator:
    """
    Tracks entry prices and computes realized PnL from actual fills.
    Replaces record_pnl_tick(0.0) placeholder.

    Called from the risk engine's on_fill hook, which is wired to the
    gateway fill notification (private WS channel).

    PnL = (exit_price - entry_price) * size * direction
    For short options: entry = vol at which we sold, exit = vol at which we buy back.
    Funding PnL is separate (from StateEngine.funding.rate_8h).
    """

    def __init__(self) -> None:
        self._entries: dict[str, tuple[float, float]] = {}  # instrument -> (price, size)
        self._realized: float = 0.0
        self._fee_total: float = 0.0
        self._fill_count: int = 0

    def record_entry(self, instrument: str, price: float, size: float) -> None:
        self._entries[instrument] = (price, size)

    def on_fill(self, instrument: str, fill_price: float, fill_size: float,
                side: str, fee_usd: float = 0.0) -> float:
        """
        Returns realized PnL for this fill.
        side: "sell" = opening short, "buy" = closing short.
        For our short gamma book, PnL crystallizes on the buy-back.
        """
        self._fee_total  += fee_usd
        self._fill_count += 1

        entry_price, entry_size = self._entries.get(instrument, (fill_price, fill_size))

        if side == "buy":
            # closing a short: we sold high, buying low = positive
            pnl = (entry_price - fill_price) * fill_size - fee_usd
        else:
            # opening a short: no realized PnL yet, just record entry
            self._entries[instrument] = (fill_price, fill_size)
            return 0.0

        self._realized += pnl
        log.info(
            f"realized PnL | {instrument} | entry={entry_price:.4f} "
            f"exit={fill_price:.4f} size={fill_size} pnl=${pnl:.2f} "
            f"cumulative=${self._realized:.2f}"
        )
        return pnl

    def funding_pnl(self, state: StateEngine, notional_usd: float, dt_h: float) -> float:
        """Funding income for the current period. Short perp = positive when rate > 0."""
        return state.funding.rate_8h * notional_usd * (dt_h / 8.0)

    @property
    def realized(self) -> float:
        return self._realized

    @property
    def fees_paid(self) -> float:
        return self._fee_total


# ---- margin from exchange ---------------------------------------------------

class MarginMonitor:
    """
    Margin utilization. Two modes:
      live:      fetched from Deribit private/get_account_summary on each risk cycle
      estimated: falls back to position sizing heuristic if gateway not available

    Deribit portfolio margin is complex (cross-margined). Don't try to replicate
    their margin formula - just read it from the API. The estimated path is a
    rough sanity check only.
    """

    def __init__(self, cfg: Config) -> None:
        m = cfg.risk.margin
        self._warn_pct   = m.margin_warning_pct
        self._reduce_pct = m.margin_reduce_pct
        self._halt_pct   = m.margin_halt_pct

        self._last_utilization: float = 0.0
        self._last_fetch_s:     float = 0.0
        self._fetch_interval_s: float = 10.0   # don't hammer the API

    async def fetch_utilization(self, gateway) -> float:
        """
        Pull actual margin utilization from Deribit.
        Returns 0.0 on failure - fail open, let other checks catch real problems.
        """
        now = time.monotonic()
        if now - self._last_fetch_s < self._fetch_interval_s:
            return self._last_utilization

        try:
            data = await gateway.get_account_summary()
            equity           = data.get("equity", 0.0)
            initial_margin   = data.get("initial_margin", 0.0)
            if equity > 0:
                utilization = initial_margin / equity
                self._last_utilization = utilization
                self._last_fetch_s     = now
                return utilization
        except Exception as e:
            log.debug(f"margin fetch failed: {e}")

        return self._last_utilization

    def check(self, utilization: float) -> tuple[RiskLevel, str]:
        if utilization >= self._halt_pct:
            return RiskLevel.HALT, f"margin {utilization:.0%} >= halt {self._halt_pct:.0%}"
        if utilization >= self._reduce_pct:
            return RiskLevel.REDUCE, f"margin {utilization:.0%} >= reduce {self._reduce_pct:.0%}"
        if utilization >= self._warn_pct:
            return RiskLevel.WARN, f"margin {utilization:.0%} nearing limit"
        return RiskLevel.OK, ""

    @property
    def last_utilization(self) -> float:
        return self._last_utilization


# ---- drawdown tracker -------------------------------------------------------

class DrawdownTracker:
    def __init__(self, cfg: Config) -> None:
        dd = cfg.risk.drawdown
        self._max_intraday_usd = dd.max_intraday_drawdown_usd
        self._max_24h_usd      = dd.max_drawdown_24h_usd
        self._max_7d_usd       = dd.max_drawdown_7d_usd
        self._max_30d_usd      = dd.max_drawdown_30d_usd
        self._recovery_usd     = dd.recovery_threshold_usd
        self._recovery_wait_h  = dd.recovery_wait_h

        self._session_high: float = 0.0
        self._halted_at_ms: int   = 0
        self._pnl_log: deque[tuple[float, float]] = deque(maxlen=30 * 24 * 60)

    def update(self, cumulative_pnl: float) -> tuple[RiskLevel, str]:
        now = time.time()
        self._pnl_log.append((now, cumulative_pnl))
        self._session_high = max(self._session_high, cumulative_pnl)

        intraday_dd = self._session_high - cumulative_pnl
        if intraday_dd >= self._max_intraday_usd:
            return RiskLevel.HALT, f"intraday drawdown ${intraday_dd:,.0f} >= limit ${self._max_intraday_usd:,.0f}"

        for window_s, limit in [
            (86_400,    self._max_24h_usd),
            (604_800,   self._max_7d_usd),
            (2_592_000, self._max_30d_usd),
        ]:
            cutoff  = now - window_s
            old_pnl = next((pnl for ts, pnl in self._pnl_log if ts >= cutoff), cumulative_pnl)
            dd      = old_pnl - cumulative_pnl
            if dd >= limit:
                label = {86_400: "24h", 604_800: "7d", 2_592_000: "30d"}[window_s]
                return RiskLevel.HALT, f"{label} drawdown ${dd:,.0f} >= limit ${limit:,.0f}"

        return RiskLevel.OK, ""

    def in_recovery(self, cumulative_pnl: float) -> bool:
        if self._halted_at_ms == 0:
            return False
        elapsed_h = (_now_ms() - self._halted_at_ms) / 3_600_000
        if elapsed_h < self._recovery_wait_h:
            return True
        pnl_at_halt = next(
            (pnl for ts, pnl in self._pnl_log if abs(ts * 1000 - self._halted_at_ms) < 5000),
            cumulative_pnl,
        )
        return (cumulative_pnl - pnl_at_halt) < self._recovery_usd

    def record_halt(self) -> None:
        self._halted_at_ms = _now_ms()

    def reset_session(self) -> None:
        self._session_high = 0.0


# ---- loss velocity ----------------------------------------------------------

class LossVelocityMonitor:
    def __init__(self, max_per_hour: float, max_per_minute: float) -> None:
        self._max_per_hour   = max_per_hour
        self._max_per_minute = max_per_minute
        self._ticks: deque[tuple[float, float]] = deque(maxlen=10_000)

    def record(self, pnl_usd: float) -> None:
        self._ticks.append((time.monotonic(), pnl_usd))

    def check(self) -> tuple[RiskLevel, str]:
        if len(self._ticks) < 2:
            return RiskLevel.OK, ""
        now = time.monotonic()
        for window_s, limit, label in [
            (60,   self._max_per_minute, "1min"),
            (3600, self._max_per_hour,   "1h"),
        ]:
            pnl_slice = [p for ts, p in self._ticks if ts >= now - window_s]
            if not pnl_slice:
                continue
            loss = -sum(pnl_slice)
            if loss >= limit:
                return RiskLevel.HALT, f"loss velocity ${loss:,.0f} in {label} >= ${limit:,.0f}"
        return RiskLevel.OK, ""


# ---- position limits --------------------------------------------------------

class PositionLimitChecker:
    def __init__(self, cfg: Config, asset: str) -> None:
        self._lim   = cfg.risk.asset_limits(asset)
        self._asset = asset

    def check(self, state: StateEngine) -> tuple[RiskLevel, str]:
        inv = state.inventory

        if abs(inv.net_vega) > self._lim.max_vega_usd:
            return RiskLevel.HALT, f"vega ${abs(inv.net_vega):,.0f} > limit"

        if abs(inv.net_gamma) > self._lim.max_gamma_usd:
            return RiskLevel.HALT, f"gamma ${abs(inv.net_gamma):,.0f} > limit"

        try:
            delta_notional = abs(inv.net_delta) * state.spot()
        except StateError:
            delta_notional = 0.0

        if delta_notional > self._lim.max_delta_notional_usd:
            return RiskLevel.HALT, f"delta notional ${delta_notional:,.0f} > limit"

        if abs(inv.perp_position_usd) > self._lim.max_perp_notional_usd:
            return RiskLevel.HALT, f"perp notional ${abs(inv.perp_position_usd):,.0f} > limit"

        return RiskLevel.OK, ""


# ---- market conditions ------------------------------------------------------

class MarketConditionChecker:
    def __init__(self, cfg: Config) -> None:
        ks = cfg.risk.kill_switch
        ob = cfg.market.orderbook
        self._rv_reduce     = ks.rv_spike_threshold
        self._rv_halt       = ks.rv_spike_halt_threshold
        self._funding_halt  = ks.funding_negative_halt_ann
        self._book_stale_ms = ob.stale_book_threshold_ms
        self._idx_div       = ks.index_divergence_pct

    def check(self, state: StateEngine) -> tuple[RiskLevel, str]:
        worst_level  = RiskLevel.OK
        worst_reason = ""

        try:
            ratio = state.rv_spike_ratio()
            if ratio >= self._rv_halt:
                return RiskLevel.HALT, f"RV spike {ratio:.2f}x >= halt {self._rv_halt}x"
            if ratio >= self._rv_reduce:
                worst_level  = RiskLevel.REDUCE
                worst_reason = f"RV spike {ratio:.2f}x >= reduce {self._rv_reduce}x"
        except StateError:
            pass

        funding_ann = state.funding.rate_ann
        if funding_ann < self._funding_halt:
            return RiskLevel.HALT, f"funding {funding_ann:.1%} < halt {self._funding_halt:.1%}"

        try:
            div = abs(state.perp_mid() - state.spot()) / state.spot()
            if div > self._idx_div:
                return RiskLevel.HALT, f"perp/index divergence {div:.2%}"
        except StateError:
            pass

        if state.perp_book.is_stale(self._book_stale_ms):
            return RiskLevel.HALT, f"perp book stale > {self._book_stale_ms}ms"

        return worst_level, worst_reason


# ---- API health -------------------------------------------------------------

class ApiHealthTracker:
    def __init__(self, cfg: Config) -> None:
        ks = cfg.risk.kill_switch
        self._max_errors     = ks.api_error_consecutive
        self._latency_ms     = ks.api_latency_threshold_ms
        self._latency_consec = ks.api_latency_consecutive
        self._ws_disconnect_s= ks.websocket_disconnect_max_s
        self._error_count    = 0
        self._slow_count     = 0
        self._last_ws_ms     = _now_ms()

    def record_error(self) -> None:
        self._error_count += 1

    def record_success(self) -> None:
        self._error_count = 0

    def record_ws_heartbeat(self) -> None:
        self._last_ws_ms = _now_ms()

    def check(self, execution: ExecutionEngine) -> tuple[RiskLevel, str]:
        if self._error_count >= self._max_errors:
            return RiskLevel.HALT, f"{self._error_count} consecutive API errors"

        p95 = execution.latency_summary().get("order_send", 0.0)
        if p95 > self._latency_ms:
            self._slow_count += 1
            if self._slow_count >= self._latency_consec:
                return RiskLevel.HALT, f"P95 latency {p95:.0f}ms for {self._slow_count} checks"
        else:
            self._slow_count = 0

        ws_age = (_now_ms() - self._last_ws_ms) / 1000.0
        if ws_age > self._ws_disconnect_s:
            return RiskLevel.HALT, f"WS silent {ws_age:.1f}s"

        return RiskLevel.OK, ""


# ---- main risk engine -------------------------------------------------------

class RiskEngine:
    """
    Independent risk monitor. Has veto over all order flow.
    Wired to real fill PnL (no more record_pnl_tick(0.0)).
    Pulls margin from exchange every 10s.
    Sends alerts via Telegram/Slack when it halts.
    Per-leg PnL breakdown in snapshot.
    """

    def __init__(
        self,
        cfg:       Config,
        state:     StateEngine,
        execution: ExecutionEngine,
        asset:     str,
        on_halt:   Callable[[KillReason, str], Awaitable[None]] | None = None,
    ) -> None:
        self.cfg       = cfg
        self.state     = state
        self.execution = execution
        self.asset     = asset
        self._on_halt  = on_halt

        self._halted       = False
        self._halt_reason: KillReason | None = None
        self._halt_time_ms = 0

        self._fill_pnl    = FillPnlAccumulator()
        self._attributor  = PnlAttributor(
            recompute_interval_s = cfg.risk.pnl_attribution.recompute_interval_s,
            dominance_threshold  = cfg.risk.pnl_attribution.component_dominance_threshold,
        )
        self._drawdown    = DrawdownTracker(cfg)
        self._velocity    = LossVelocityMonitor(
            max_per_hour   = cfg.risk.kill_switch.max_loss_per_hour_usd,
            max_per_minute = cfg.risk.kill_switch.max_loss_per_minute_usd,
        )
        self._pos_limits  = PositionLimitChecker(cfg, asset)
        self._market      = MarketConditionChecker(cfg)
        self._api_health  = ApiHealthTracker(cfg)
        self._margin      = MarginMonitor(cfg)
        self._alerter     = Alerter()

        self._check_interval_s = 1.0
        self._log_interval_s   = 60.0
        self._last_log         = 0.0
        self._running          = False

        log.info(f"risk engine ready | asset={asset} alerting={self._alerter._enabled}")

    # ---- public interface ---------------------------------------------------

    @property
    def is_halted(self) -> bool:
        return self._halted

    def on_fill(
        self,
        instrument:  str,
        fill_price:  float,
        fill_size:   float,
        side:        str,
        fee_usd:     float = 0.0,
    ) -> None:
        """
        Called on every fill. Computes real PnL and records to attributor.
        Replaces the record_pnl_tick(0.0) placeholder entirely.
        """
        pnl = self._fill_pnl.on_fill(instrument, fill_price, fill_size, side, fee_usd)
        self._velocity.record(pnl)
        self._attributor.record_fill(instrument, fill_price, fill_size, side)

        total = self._fill_pnl.realized + self._funding_income_since_start()
        self._drawdown.update(total)   # update drawdown on every fill, not just timer

    def on_entry(self, instrument: str, entry_price: float, size: float) -> None:
        """Record entry price so we can compute realized PnL on close."""
        self._fill_pnl.record_entry(instrument, entry_price, size)
        self._attributor.record_entry(instrument, entry_price)

    def record_api_error(self) -> None:
        self._api_health.record_error()

    def record_api_success(self) -> None:
        self._api_health.record_success()

    def record_ws_heartbeat(self) -> None:
        self._api_health.record_ws_heartbeat()

    def update_pnl_components(
        self, spot: float, iv: float, gamma: float, theta: float,
        vega: float, funding_8h: float, notional: float, fees_paid: float,
    ) -> PnlComponents:
        return self._attributor.update(spot, iv, gamma, theta, vega, funding_8h, notional, fees_paid)

    async def manual_halt(self, reason: str = "manual") -> None:
        await self._trigger_halt(KillReason.MANUAL, reason)

    async def resume(self) -> None:
        if not self._halted:
            return
        total = self._fill_pnl.realized + self._funding_income_since_start()
        if self._drawdown.in_recovery(total):
            log.warning("resume rejected - still in recovery")
            return
        log.info("risk engine resuming")
        self._halted       = False
        self._halt_reason  = None
        self._halt_time_ms = 0

    # ---- main loop ----------------------------------------------------------

    async def run(self) -> None:
        self._running = True
        log.info("risk engine loop started")

        while self._running:
            try:
                await self._run_checks()
                await self._maybe_log()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"risk check error: {e}", exc_info=True)

            await asyncio.sleep(self._check_interval_s)

    async def stop(self) -> None:
        self._running = False

    # ---- checks -------------------------------------------------------------

    async def _run_checks(self) -> None:
        if self._halted:
            return

        # margin from exchange (async, runs concurrently with other checks)
        margin_util = await self._margin.fetch_utilization(self.execution.gateway)

        results = await asyncio.gather(
            self._check_market_conditions(),
            self._check_position_limits(),
            self._check_drawdown(),
            self._check_loss_velocity(),
            self._check_api_health(),
            self._check_margin(margin_util),
        )

        for level, reason, kill_reason in results:
            if level == RiskLevel.HALT:
                await self._trigger_halt(kill_reason, reason)
                return
            if level == RiskLevel.REDUCE:
                log.warning(f"REDUCE | {reason}")
            if level == RiskLevel.WARN:
                await self._alerter.warn(self.asset, reason)

    async def _check_market_conditions(self) -> tuple[RiskLevel, str, KillReason]:
        level, reason = self._market.check(self.state)
        return level, reason, _market_kill_reason(reason)

    async def _check_position_limits(self) -> tuple[RiskLevel, str, KillReason]:
        level, reason = self._pos_limits.check(self.state)
        return level, reason, KillReason.POSITION_LIMIT

    async def _check_drawdown(self) -> tuple[RiskLevel, str, KillReason]:
        total = self._fill_pnl.realized + self._funding_income_since_start()
        level, reason = self._drawdown.update(total)
        return level, reason, _drawdown_kill_reason(reason)

    async def _check_loss_velocity(self) -> tuple[RiskLevel, str, KillReason]:
        level, reason = self._velocity.check()
        return level, reason, KillReason.LOSS_VELOCITY

    async def _check_api_health(self) -> tuple[RiskLevel, str, KillReason]:
        level, reason = self._api_health.check(self.execution)
        kr = KillReason.API_LATENCY if "latency" in reason.lower() else KillReason.API_ERRORS
        return level, reason, kr

    async def _check_margin(self, utilization: float) -> tuple[RiskLevel, str, KillReason]:
        level, reason = self._margin.check(utilization)
        return level, reason, KillReason.MARGIN_BREACH

    def _funding_income_since_start(self) -> float:
        """Approximate funding income. Real accounting uses fills."""
        try:
            notional = abs(self.state.inventory.perp_position_usd)
            rate_8h  = self.state.funding.rate_8h
            # rough: sessions don't track start time, use a proxy
            return rate_8h * notional * 0.1   # placeholder until proper funding ledger
        except Exception:
            return 0.0

    # ---- halt ---------------------------------------------------------------

    async def _trigger_halt(self, reason: KillReason, detail: str) -> None:
        if self._halted:
            return

        self._halted       = True
        self._halt_reason  = reason
        self._halt_time_ms = _now_ms()
        self._drawdown.record_halt()

        # alert first (fast), then flatten (slow)
        await self._alerter.halt(self.asset, reason, detail)

        try:
            await self.execution._emergency_flatten(reason=f"{reason.name} | {detail}")
        except Exception as e:
            log.error(f"flatten failed: {e}")

        if self._on_halt:
            try:
                await self._on_halt(reason, detail)
            except Exception as e:
                log.error(f"on_halt callback: {e}")

    # ---- logging ------------------------------------------------------------

    async def _maybe_log(self) -> None:
        now = time.monotonic()
        if now - self._last_log < self._log_interval_s:
            return
        self._last_log = now
        log.info(f"risk | {self.snapshot()}")

    def snapshot(self) -> dict:
        try:
            rv_ratio = self.state.rv_spike_ratio()
        except StateError:
            rv_ratio = 0.0

        return {
            "asset":          self.asset,
            "halted":         self._halted,
            "halt_reason":    self._halt_reason.name if self._halt_reason else None,
            "realized_pnl":   self._fill_pnl.realized,
            "fees_paid":      self._fill_pnl.fees_paid,
            "pnl_components": self._attributor.latest.as_dict(),
            "rv_ratio":       rv_ratio,
            "funding_ann":    self.state.funding.rate_ann,
            "margin_util":    self._margin.last_utilization,
            "net_vega":       self.state.inventory.net_vega,
            "net_gamma":      self.state.inventory.net_gamma,
            "live_orders":    self.execution.live_order_count(),
            "latency_p95":    self.execution.latency_summary(),
        }


# ---- helpers ----------------------------------------------------------------

def _market_kill_reason(reason: str) -> KillReason:
    if "RV spike"   in reason: return KillReason.RV_SPIKE
    if "funding"    in reason: return KillReason.FUNDING_NEGATIVE
    if "divergence" in reason: return KillReason.INDEX_DIVERGENCE
    if "stale"      in reason: return KillReason.BOOK_STALE
    return KillReason.MANUAL


def _drawdown_kill_reason(reason: str) -> KillReason:
    if "intraday" in reason: return KillReason.DRAWDOWN_INTRADAY
    if "24h"      in reason: return KillReason.DRAWDOWN_24H
    if "7d"       in reason: return KillReason.DRAWDOWN_7D
    return KillReason.DRAWDOWN_INTRADAY


def _now_ms() -> int:
    return int(time.monotonic() * 1000)

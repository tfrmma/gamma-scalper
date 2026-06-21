from __future__ import annotations

import asyncio
import logging
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
    OK      = auto()
    WARN    = auto()   # log + alert, no action
    REDUCE  = auto()   # cut position size
    HALT    = auto()   # flatten everything, stop trading


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


# ---- PnL attribution --------------------------------------------------------

@dataclass
class PnlComponents:
    """
    Decomposed PnL. Updated every recompute_interval_s.
    Numbers are approximate - good enough for monitoring, not for accounting.

    gamma_pnl  = 0.5 * Γ * ΔS²   (what we're trying to capture / avoid)
    theta_pnl  = Θ * Δt           (time decay - negative for long gamma, positive for short)
    vega_pnl   = ν * Δσ           (vol change exposure - residual, should be small)
    funding    = interest_8h * notional  (income from short perp)
    tx_cost    = fees paid this period
    """
    gamma_pnl:  float = 0.0
    theta_pnl:  float = 0.0
    vega_pnl:   float = 0.0
    funding:    float = 0.0
    tx_cost:    float = 0.0

    @property
    def total(self) -> float:
        return self.gamma_pnl + self.theta_pnl + self.vega_pnl + self.funding - self.tx_cost

    def dominant_component(self) -> tuple[str, float]:
        components = {
            "gamma":   abs(self.gamma_pnl),
            "theta":   abs(self.theta_pnl),
            "vega":    abs(self.vega_pnl),
            "funding": abs(self.funding),
            "tx_cost": abs(self.tx_cost),
        }
        name = max(components, key=components.get)
        total_abs = sum(components.values())
        ratio = components[name] / total_abs if total_abs > 0 else 0.0
        return name, ratio

    def as_dict(self) -> dict:
        return {
            "gamma_pnl": self.gamma_pnl,
            "theta_pnl": self.theta_pnl,
            "vega_pnl":  self.vega_pnl,
            "funding":   self.funding,
            "tx_cost":   self.tx_cost,
            "total":     self.total,
        }


class PnlAttributor:
    """
    Approximate PnL decomposition from greeks + market moves.
    Not trying to be exact here - if you want exact, reconcile against exchange statements.
    This is for real-time monitoring only.
    """

    def __init__(self, recompute_interval_s: int, dominance_threshold: float) -> None:
        self._interval   = recompute_interval_s
        self._dominance  = dominance_threshold
        self._last_run   = 0.0
        self._last_spot  = 0.0
        self._last_iv    = 0.0
        self._components = PnlComponents()
        self._history: deque[tuple[float, PnlComponents]] = deque(maxlen=1440)  # 24h at 1min

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

            self._components.gamma_pnl  = 0.5 * gamma * (dS ** 2)
            self._components.theta_pnl  = theta * dt_h            # theta is per hour
            self._components.vega_pnl   = vega  * dv
            self._components.funding    = funding_8h * notional * (dt_h / 8.0)
            self._components.tx_cost    = fees_paid

        self._last_spot = spot
        self._last_iv   = iv
        self._last_run  = now
        self._history.append((now, self._components))

        name, ratio = self._components.dominant_component()
        if ratio > self._dominance:
            log.warning(f"PnL dominated by {name} ({ratio:.0%}) - check model")

        return self._components

    @property
    def latest(self) -> PnlComponents:
        return self._components


# ---- drawdown tracker -------------------------------------------------------

class DrawdownTracker:
    """
    Rolling drawdown over multiple windows.
    Uses a high-water mark per window - straightforward, hard to get wrong.
    """

    def __init__(self, cfg: Config) -> None:
        dd = cfg.risk.drawdown
        self._max_intraday_usd = dd.max_intraday_drawdown_usd
        self._max_intraday_pct = dd.max_intraday_drawdown_pct
        self._max_24h_usd      = dd.max_drawdown_24h_usd
        self._max_7d_usd       = dd.max_drawdown_7d_usd
        self._max_30d_usd      = dd.max_drawdown_30d_usd
        self._recovery_usd     = dd.recovery_threshold_usd
        self._recovery_wait_h  = dd.recovery_wait_h

        self._session_high: float = 0.0
        self._session_start_pnl: float = 0.0
        self._halted_at_ms: int   = 0

        # rolling pnl snapshots: (timestamp_s, cumulative_pnl)
        self._pnl_log: deque[tuple[float, float]] = deque(maxlen=30 * 24 * 60)

    def update(self, cumulative_pnl: float) -> tuple[RiskLevel, str]:
        now = time.time()
        self._pnl_log.append((now, cumulative_pnl))
        self._session_high = max(self._session_high, cumulative_pnl)

        # intraday
        intraday_dd = self._session_high - cumulative_pnl
        if intraday_dd >= self._max_intraday_usd:
            return RiskLevel.HALT, f"intraday drawdown ${intraday_dd:,.0f} >= limit ${self._max_intraday_usd:,.0f}"

        # rolling windows
        for window_s, limit in [
            (86_400,   self._max_24h_usd),
            (604_800,  self._max_7d_usd),
            (2_592_000, self._max_30d_usd),
        ]:
            cutoff = now - window_s
            old_pnl = next(
                (pnl for ts, pnl in self._pnl_log if ts >= cutoff),
                cumulative_pnl,
            )
            window_dd = old_pnl - cumulative_pnl   # negative = drawdown
            if window_dd >= limit:
                label = {86_400: "24h", 604_800: "7d", 2_592_000: "30d"}[window_s]
                return RiskLevel.HALT, f"{label} drawdown ${window_dd:,.0f} >= limit ${limit:,.0f}"

        return RiskLevel.OK, ""

    def in_recovery(self, cumulative_pnl: float) -> bool:
        """True if we halted and haven't recovered enough to resume."""
        if self._halted_at_ms == 0:
            return False
        elapsed_h = (_now_ms() - self._halted_at_ms) / 3_600_000
        if elapsed_h < self._recovery_wait_h:
            return True
        # check if we've earned back recovery_threshold
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
    """
    Checks if we're losing money faster than the thresholds.
    Simple sliding window on realized PnL ticks.
    Catches runaway scenarios that drawdown limits might miss (e.g. rapid fill + adverse move).
    """

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
            cutoff    = now - window_s
            pnl_slice = [p for ts, p in self._ticks if ts >= cutoff]
            if not pnl_slice:
                continue
            loss = -sum(pnl_slice)   # sum of losses (positive = losing)
            if loss >= limit:
                return RiskLevel.HALT, f"loss velocity ${loss:,.0f} in {label} >= limit ${limit:,.0f}"

        return RiskLevel.OK, ""


# ---- position limit checker -------------------------------------------------

class PositionLimitChecker:
    def __init__(self, cfg: Config, asset: str) -> None:
        self._lim  = cfg.risk.asset_limits(asset)
        self._agg  = cfg.risk.aggregate_limits
        self._asset = asset

    def check(self, state: StateEngine) -> tuple[RiskLevel, str]:
        inv = state.inventory

        if abs(inv.net_vega) > self._lim.max_vega_usd:
            return RiskLevel.HALT, f"vega ${abs(inv.net_vega):,.0f} > limit ${self._lim.max_vega_usd:,.0f}"

        if abs(inv.net_gamma) > self._lim.max_gamma_usd:
            return RiskLevel.HALT, f"gamma ${abs(inv.net_gamma):,.0f} > limit ${self._lim.max_gamma_usd:,.0f}"

        try:
            delta_notional = abs(inv.net_delta) * state.spot()
        except StateError:
            delta_notional = 0.0

        if delta_notional > self._lim.max_delta_notional_usd:
            return RiskLevel.HALT, f"delta notional ${delta_notional:,.0f} > limit ${self._lim.max_delta_notional_usd:,.0f}"

        if abs(inv.perp_position_usd) > self._lim.max_perp_notional_usd:
            return RiskLevel.HALT, f"perp notional ${abs(inv.perp_position_usd):,.0f} > limit"

        return RiskLevel.OK, ""


# ---- market condition checks ------------------------------------------------

class MarketConditionChecker:
    """
    RV spike, funding, book health, index divergence.
    These are the checks that run on every tick regardless of position state.
    """

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

        # RV spike - corr(rv, premium) = -0.928, this is the most important check
        try:
            ratio = state.rv_spike_ratio()
            if ratio >= self._rv_halt:
                return RiskLevel.HALT, f"RV spike ratio {ratio:.2f}x >= halt threshold {self._rv_halt}x"
            if ratio >= self._rv_reduce:
                worst_level  = RiskLevel.REDUCE
                worst_reason = f"RV spike ratio {ratio:.2f}x >= reduce threshold {self._rv_reduce}x"
        except StateError:
            pass

        # funding - always check, independent of RV
        funding_ann = state.funding.rate_ann
        if funding_ann < self._funding_halt:
            return RiskLevel.HALT, f"funding {funding_ann:.1%} ann < halt threshold {self._funding_halt:.1%}"

        # perp vs index divergence
        try:
            perp_mid   = state.perp_mid()
            index_px   = state.spot()
            divergence = abs(perp_mid - index_px) / index_px
            if divergence > self._idx_div:
                return RiskLevel.HALT, f"perp/index divergence {divergence:.2%} > {self._idx_div:.2%}"
        except StateError:
            pass

        # book staleness
        if state.perp_book.is_stale(self._book_stale_ms):
            return RiskLevel.HALT, f"perp book stale > {self._book_stale_ms}ms"

        return worst_level, worst_reason


# ---- API health tracker -----------------------------------------------------

class ApiHealthTracker:
    """
    Consecutive error counter + latency monitor.
    Execution engine records latency, we read it here.
    """

    def __init__(self, cfg: Config) -> None:
        ks = cfg.risk.kill_switch
        self._max_errors      = ks.api_error_consecutive
        self._latency_ms      = ks.api_latency_threshold_ms
        self._latency_consec  = ks.api_latency_consecutive
        self._ws_disconnect_s = ks.websocket_disconnect_max_s

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

        # check P95 send latency from execution engine
        p95_send = execution.latency_summary().get("order_send", 0.0)
        if p95_send > self._latency_ms:
            self._slow_count += 1
            if self._slow_count >= self._latency_consec:
                return RiskLevel.HALT, f"P95 order latency {p95_send:.0f}ms > {self._latency_ms}ms for {self._slow_count} checks"
        else:
            self._slow_count = 0

        ws_age_s = (_now_ms() - self._last_ws_ms) / 1000.0
        if ws_age_s > self._ws_disconnect_s:
            return RiskLevel.HALT, f"WS silent for {ws_age_s:.1f}s > {self._ws_disconnect_s}s"

        return RiskLevel.OK, ""


# ---- margin monitor ---------------------------------------------------------

class MarginMonitor:
    def __init__(self, cfg: Config) -> None:
        m = cfg.risk.margin
        self._warn_pct   = m.margin_warning_pct
        self._reduce_pct = m.margin_reduce_pct
        self._halt_pct   = m.margin_halt_pct

    def check(self, utilization: float) -> tuple[RiskLevel, str]:
        if utilization >= self._halt_pct:
            return RiskLevel.HALT, f"margin utilization {utilization:.0%} >= halt {self._halt_pct:.0%}"
        if utilization >= self._reduce_pct:
            return RiskLevel.REDUCE, f"margin utilization {utilization:.0%} >= reduce threshold"
        if utilization >= self._warn_pct:
            return RiskLevel.WARN, f"margin utilization {utilization:.0%} - approaching limit"
        return RiskLevel.OK, ""


# ---- main risk engine -------------------------------------------------------

class RiskEngine:
    """
    Runs on its own asyncio loop, independent of the strategy and execution layers.
    Has veto power over all trading activity via the halt flag.

    When halted: execution engine won't send new orders (it checks is_halted before handling signals).
    Flatten happens immediately via execution._emergency_flatten.

    TODO: add proper alerting (PagerDuty / Telegram) - right now it just logs.
          before going live, you want to wake up when this fires.
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

        self._cumulative_pnl = 0.0

        self._drawdown    = DrawdownTracker(cfg)
        self._velocity    = LossVelocityMonitor(
            max_per_hour   = cfg.risk.kill_switch.max_loss_per_hour_usd,
            max_per_minute = cfg.risk.kill_switch.max_loss_per_minute_usd,
        )
        self._pos_limits  = PositionLimitChecker(cfg, asset)
        self._market      = MarketConditionChecker(cfg)
        self._api_health  = ApiHealthTracker(cfg)
        self._margin      = MarginMonitor(cfg)
        self._attributor  = PnlAttributor(
            recompute_interval_s = cfg.risk.pnl_attribution.recompute_interval_s,
            dominance_threshold  = cfg.risk.pnl_attribution.component_dominance_threshold,
        )

        self._check_interval_s = 1.0   # run checks every second
        self._log_interval_s   = 60.0
        self._last_log         = 0.0
        self._running          = False

        log.info(f"risk engine ready | asset={asset}")

    # ---- public interface ---------------------------------------------------

    @property
    def is_halted(self) -> bool:
        return self._halted

    def record_pnl_tick(self, pnl_delta_usd: float) -> None:
        """Call on every realized PnL event (fill, funding payment, etc.)."""
        self._cumulative_pnl += pnl_delta_usd
        self._velocity.record(pnl_delta_usd)

    def record_api_error(self) -> None:
        self._api_health.record_error()

    def record_api_success(self) -> None:
        self._api_health.record_success()

    def record_ws_heartbeat(self) -> None:
        self._api_health.record_ws_heartbeat()

    def update_pnl_components(
        self,
        spot: float,
        iv: float,
        gamma: float,
        theta: float,
        vega: float,
        funding_8h: float,
        notional: float,
        fees_paid: float,
    ) -> PnlComponents:
        return self._attributor.update(spot, iv, gamma, theta, vega, funding_8h, notional, fees_paid)

    async def manual_halt(self, reason: str = "manual") -> None:
        """Operator-triggered halt. Bypasses all checks."""
        log.warning(f"MANUAL HALT requested: {reason}")
        await self._trigger_halt(KillReason.MANUAL, reason)

    async def resume(self) -> None:
        """
        Attempt to resume after a halt.
        Recovery conditions must be met - checked inside.
        Don't call this without understanding why we halted.
        """
        if not self._halted:
            return
        if self._drawdown.in_recovery(self._cumulative_pnl):
            log.warning("resume rejected - still in drawdown recovery period")
            return
        log.info("risk engine resuming")
        self._halted      = False
        self._halt_reason = None
        self._halt_time_ms = 0

    # ---- main loop ----------------------------------------------------------

    async def run(self) -> None:
        """
        Main risk loop. Call this as an asyncio task alongside the strategy loop.
        Runs until cancelled.
        """
        self._running = True
        log.info("risk engine loop started")

        while self._running:
            try:
                await self._run_checks()
                await self._maybe_log()
            except asyncio.CancelledError:
                log.info("risk engine cancelled")
                break
            except Exception as e:
                # don't let a check bug kill the monitor - but do log it loudly
                log.error(f"risk check error (non-fatal): {e}", exc_info=True)

            await asyncio.sleep(self._check_interval_s)

    async def stop(self) -> None:
        self._running = False

    # ---- check dispatch -----------------------------------------------------

    async def _run_checks(self) -> None:
        if self._halted:
            return   # already halted, nothing to check

        checks = [
            self._check_market_conditions(),
            self._check_position_limits(),
            self._check_drawdown(),
            self._check_loss_velocity(),
            self._check_api_health(),
        ]
        # run all checks, take the worst result
        # don't short-circuit - want to log everything that's wrong
        results = await asyncio.gather(*checks)

        for level, reason, kill_reason in results:
            if level == RiskLevel.HALT:
                await self._trigger_halt(kill_reason, reason)
                return   # halt fires, stop processing
            if level == RiskLevel.REDUCE:
                log.warning(f"REDUCE signal: {reason}")
                # reduction is advisory - strategy layer picks it up via state
                # TODO: push reduce signal to strategy more directly

    async def _check_market_conditions(self) -> tuple[RiskLevel, str, KillReason]:
        level, reason = self._market.check(self.state)
        kr = _market_kill_reason(reason)
        return level, reason, kr

    async def _check_position_limits(self) -> tuple[RiskLevel, str, KillReason]:
        level, reason = self._pos_limits.check(self.state)
        return level, reason, KillReason.POSITION_LIMIT

    async def _check_drawdown(self) -> tuple[RiskLevel, str, KillReason]:
        level, reason = self._drawdown.update(self._cumulative_pnl)
        if level == RiskLevel.HALT:
            return level, reason, _drawdown_kill_reason(reason)
        return level, reason, KillReason.DRAWDOWN_INTRADAY

    async def _check_loss_velocity(self) -> tuple[RiskLevel, str, KillReason]:
        level, reason = self._velocity.check()
        return level, reason, KillReason.LOSS_VELOCITY

    async def _check_api_health(self) -> tuple[RiskLevel, str, KillReason]:
        level, reason = self._api_health.check(self.execution)
        kr = KillReason.API_LATENCY if "latency" in reason.lower() else KillReason.API_ERRORS
        return level, reason, kr

    # ---- halt ---------------------------------------------------------------

    async def _trigger_halt(self, reason: KillReason, detail: str) -> None:
        if self._halted:
            return   # already halted, don't double-fire

        self._halted       = True
        self._halt_reason  = reason
        self._halt_time_ms = _now_ms()
        self._drawdown.record_halt()

        log.error(f"KILL SWITCH | {reason.name} | {detail}")

        # flatten first, notify second
        try:
            await self.execution._emergency_flatten(reason=f"risk: {reason.name} | {detail}")
        except Exception as e:
            log.error(f"flatten failed after kill switch: {e}")

        if self._on_halt:
            try:
                await self._on_halt(reason, detail)
            except Exception as e:
                log.error(f"on_halt callback error: {e}")

    # ---- logging ------------------------------------------------------------

    async def _maybe_log(self) -> None:
        now = time.monotonic()
        if now - self._last_log < self._log_interval_s:
            return
        self._last_log = now
        log.info(f"risk snapshot | {self.snapshot()}")

    def snapshot(self) -> dict:
        try:
            rv_ratio = self.state.rv_spike_ratio()
        except StateError:
            rv_ratio = 0.0

        pnl = self._attributor.latest

        return {
            "asset":          self.asset,
            "halted":         self._halted,
            "halt_reason":    self._halt_reason.name if self._halt_reason else None,
            "cumulative_pnl": self._cumulative_pnl,
            "pnl_components": pnl.as_dict(),
            "rv_ratio":       rv_ratio,
            "funding_ann":    self.state.funding.rate_ann,
            "net_vega":       self.state.inventory.net_vega,
            "net_gamma":      self.state.inventory.net_gamma,
            "live_orders":    self.execution.live_order_count(),
            "latency_p95":    self.execution.latency_summary(),
        }


# ---- helpers ----------------------------------------------------------------

def _market_kill_reason(reason: str) -> KillReason:
    if "RV spike" in reason:
        return KillReason.RV_SPIKE
    if "funding" in reason:
        return KillReason.FUNDING_NEGATIVE
    if "divergence" in reason:
        return KillReason.INDEX_DIVERGENCE
    if "stale" in reason:
        return KillReason.BOOK_STALE
    return KillReason.MANUAL


def _drawdown_kill_reason(reason: str) -> KillReason:
    if "intraday" in reason:
        return KillReason.DRAWDOWN_INTRADAY
    if "24h" in reason:
        return KillReason.DRAWDOWN_24H
    if "7d" in reason:
        return KillReason.DRAWDOWN_7D
    return KillReason.DRAWDOWN_INTRADAY


def _now_ms() -> int:
    return int(time.monotonic() * 1000)

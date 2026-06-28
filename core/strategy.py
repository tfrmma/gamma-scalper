from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import NamedTuple

from config.loader import Config
from core.state_engine import StateEngine, StateError, FundingRegime

log = logging.getLogger("strategy")


# ---- output types -----------------------------------------------------------

class StrategyAction(Enum):
    ENTER    = auto()   # open straddle
    HOLD     = auto()   # nothing to do
    HEDGE    = auto()   # delta hedge only, no position change
    REDUCE   = auto()   # trim position, premium compressing
    EXIT     = auto()   # close everything cleanly
    FLATTEN  = auto()   # get flat now, emergency path
    ROLL     = auto()   # close current expiry, reopen next one


@dataclass(frozen=True)
class Quote:
    """Option quote computed by the AS pricer."""
    bid_vol:     float   # vol pts - sell at this (we're short gamma)
    ask_vol:     float
    mid_vol:     float
    spread_vol:  float
    reserve_vol: float   # reservation price (inventory-adjusted mid)
    skew:        float   # inventory skew applied (vol pts)


@dataclass(frozen=True)
class HedgeOrder:
    """Delta hedge to execute on the perp."""
    instrument:   str
    side:         str    # "buy" | "sell"
    notional_usd: float
    reason:       str    # for the log


@dataclass(frozen=True)
class StrategySignal:
    """What the strategy wants to do this tick. Execution layer decides how."""
    action:      StrategyAction
    asset:       str
    quote:       Quote | None        = None
    hedge:       HedgeOrder | None   = None
    target_notional_usd: float       = 0.0
    reason:      str                 = ""


# ---- Avellaneda-Stoikov pricer ----------------------------------------------

class ASPricer:
    """
    A&S 2008, adapted for options vol space, with SABR skew correction.

    Base model:
      reservation_vol = mid_vol + skew
      spread_vol      = gamma*sigma^2*T + (2/gamma)*ln(1 + gamma/k)

    SABR extension:
      When SABR is calibrated, we apply a vanna correction to the reservation price.
      Vanna (dSigma/dF) tells us how much the IV will move when spot moves - this
      affects the true cost of holding delta risk. Intuitively: if you're short a call
      and spot rips, IV also goes up (positive vanna on calls), so your real exposure
      is larger than Black-Scholes delta alone suggests.

      skew_sabr = vanna * net_delta_exposure * spot_move_1sigma
      reservation_vol += skew_sabr

    MLE k calibration:
      Poisson inter-arrival MLE: k_hat = n / sum(intervals)
      This is the exact MLE for exponential inter-arrivals, not the mean proxy.
      Exponential(lambda) has MLE lambda_hat = n / sum(t_i).
      We decay towards prior k to avoid overfitting on thin windows.
    """

    def __init__(self, cfg: Config) -> None:
        as_cfg = cfg.strategy.avellaneda_stoikov
        self.gamma_as           = as_cfg.gamma     # renamed to avoid clash with SABR gamma
        self.k_prior            = as_cfg.k
        self.T_default_h        = as_cfg.T_hours_default
        self.spread_min         = as_cfg.spread_min_vol_pts
        self.spread_max         = as_cfg.spread_max_vol_pts
        self.skew_cap           = as_cfg.skew_cap_vol_pts
        self._k_mle             = as_cfg.k
        self._k_decay           = 0.1    # weight on prior vs MLE, tune in live
        self._last_calibration  = 0.0
        self._n_calibrations    = 0

    def quote(
        self,
        mid_vol:    float,
        rv:         float,
        dte_hours:  float,
        inventory:  float,
        vanna:      float = 0.0,   # SABR dSigma/dF - 0 falls back to pure A&S
        spot:       float = 0.0,   # needed for vanna adjustment
    ) -> Quote:
        T = dte_hours / 8760.0

        # base inventory skew
        skew = self.gamma_as * (rv ** 2) * T * inventory

        # SABR vanna correction: if vol moves with spot, reservation price shifts
        # vanna > 0 means IV rises when spot rises - being short calls + long spot hedge
        # means your effective vega exposure is amplified. Widen the quote.
        if abs(vanna) > 0 and spot > 0:
            rv_1sigma   = rv * math.sqrt(1.0 / 8760.0)   # 1h 1-sigma move
            spot_move   = spot * rv_1sigma
            skew_sabr   = vanna * spot_move * abs(inventory) * 0.1   # 0.1 = dampening
            skew += skew_sabr

        skew    = max(-self.skew_cap, min(self.skew_cap, skew))
        reserve = mid_vol + skew

        vol_term     = self.gamma_as * (rv ** 2) * T
        arrival_term = (2.0 / self.gamma_as) * math.log(1.0 + self.gamma_as / self._k_mle)
        spread = max(self.spread_min, min(self.spread_max, vol_term + arrival_term))

        half = spread / 2.0
        return Quote(
            bid_vol    = reserve - half,
            ask_vol    = reserve + half,
            mid_vol    = mid_vol,
            spread_vol = spread,
            reserve_vol= reserve,
            skew       = skew,
        )

    def recalibrate_k(self, trade_intervals: list[float]) -> None:
        """
        Bayesian update of Poisson arrival rate k.

        Likelihood: inter-arrivals ~ Exponential(k)
        Prior:      k ~ Gamma(alpha_0, beta_0)  (conjugate prior)
        Posterior:  k ~ Gamma(alpha_0 + n, beta_0 + sum(intervals))

        Posterior mean = (alpha_0 + n) / (beta_0 + sum(intervals))

        With few trades (n << alpha_0) the prior dominates.
        With many trades (n >> alpha_0) the MLE dominates.
        This correctly handles thin weekends without the ad-hoc decay factor.

        Prior calibration: alpha_0=5, beta_0=3.33 gives prior mean k=1.5
        which matches k_prior from config. At n=50 trades the posterior
        is ~90% driven by data.
        """
        if not trade_intervals:
            return
        n          = len(trade_intervals)
        total_time = sum(trade_intervals)
        if total_time <= 0:
            return

        # Gamma conjugate prior: E[k] = alpha/beta = k_prior
        # set beta_0 = alpha_0 / k_prior so prior mean = k_prior
        alpha_0 = 5.0
        beta_0  = alpha_0 / self.k_prior

        # posterior mean
        self._k_mle          = (alpha_0 + n) / (beta_0 + total_time)
        self._last_calibration = time.monotonic()
        self._n_calibrations  += 1

        posterior_weight = n / (alpha_0 + n)
        log.info(
            f"k recalibrated (Bayes) | k={self._k_mle:.4f} "
            f"n={n} data_weight={posterior_weight:.0%} "
            f"calibration={self._n_calibrations}"
        )

    @property
    def k_current(self) -> float:
        return self._k_mle


# ---- position sizer ---------------------------------------------------------

class Sizer:
    """
    Computes target notional given signal strength, funding regime, and limits.
    Nothing fancy - scale base notional by regime multiplier and vol premium.
    """

    def __init__(self, cfg: Config, asset: str) -> None:
        leg  = cfg.strategy.leg[asset]
        self.base    = leg.base_notional_usd
        self.max_usd = leg.max_notional_usd
        self.min_usd = leg.min_notional_usd

    def target_notional(
        self,
        vol_premium:      float,   # IV - RV, annualized
        size_multiplier:  float,   # from FundingState
        entry_threshold:  float,
    ) -> float:
        if vol_premium < entry_threshold:
            return 0.0

        # scale up linearly as premium grows above threshold
        # cap at 2x entry threshold for full size - don't extrapolate forever
        premium_scale = min((vol_premium - entry_threshold) / entry_threshold, 1.0)
        notional = self.base * size_multiplier * (0.5 + 0.5 * premium_scale)

        return max(self.min_usd, min(self.max_usd, notional))


# ---- roll detector ----------------------------------------------------------

class RollDetector:
    """
    Three conditions that trigger a roll. Any one is sufficient.
    Not doing anything smart here - just checking thresholds.
    """

    def __init__(self, cfg: Config) -> None:
        r = cfg.strategy.rolling
        self.dte_threshold     = r.roll_on_dte_below
        self.moneyness_max     = r.roll_on_moneyness_pct
        self.surface_shift_max = r.roll_on_vol_surface_shift_pct

    def should_roll(
        self,
        dte_hours:        float,
        spot:             float,
        strike:           float,
        iv_now:           float,
        iv_at_entry:      float,
    ) -> tuple[bool, str]:
        dte_days = dte_hours / 24.0

        if dte_days < self.dte_threshold:
            return True, f"DTE={dte_days:.1f}d < {self.dte_threshold}d"

        moneyness = abs(spot - strike) / spot
        if moneyness > self.moneyness_max:
            return True, f"moneyness={moneyness:.1%} > {self.moneyness_max:.1%}"

        if iv_at_entry > 0:
            surface_shift = abs(iv_now - iv_at_entry) / iv_at_entry
            if surface_shift > self.surface_shift_max:
                return True, f"vol surface shifted {surface_shift:.1%}"

        return False, ""


# ---- hedge calculator -------------------------------------------------------

def compute_hedge(
    state:      StateEngine,
    asset:      str,
    cfg:        Config,
) -> HedgeOrder | None:
    """
    Computes the perp hedge needed to flatten delta.

    Uses delta_tracker.accumulated_delta, NOT inventory.net_delta.
    inventory.net_delta only updates on fills — it's the delta at entry,
    which for an ATM straddle is ~0 and stays 0 forever between fills.
    accumulated_delta tracks the actual drift: gamma * dS on every tick.

    Sign convention:
      accumulated_delta > 0 means we've drifted long delta (spot moved up,
      calls gained more delta than puts lost). Sell perp to flatten.
      accumulated_delta < 0: spot moved down, drifted short. Buy perp.
    """
    if not state.needs_hedge():
        return None

    accumulated  = state.delta_tracker.accumulated_delta
    spot         = state.spot()
    contract_sz  = cfg.market.assets[asset].contract_size

    # accumulated is in option delta units, convert to USD notional
    notional_usd = abs(accumulated) * spot * contract_sz

    if notional_usd < cfg.market.assets[asset].min_trade_amount_perp:
        return None

    side = "sell" if accumulated > 0 else "buy"

    return HedgeOrder(
        instrument   = cfg.market.assets[asset].perp_instrument,
        side         = side,
        notional_usd = notional_usd,
        reason       = f"accumulated_delta={accumulated:.4f} spot={spot:.0f}",
    )


# ---- main strategy ----------------------------------------------------------

@dataclass
class _Leg:
    """One expiry/strike position."""
    expiry_ts:     int
    strike:        float
    iv_at_entry:   float
    entry_time_ms: int
    notional_usd:  float

    def dte_hours(self) -> float:
        remaining_ms = self.expiry_ts * 1000 - int(time.time() * 1000)
        return max(0.0, remaining_ms / 3_600_000)


class PositionState:
    """
    Registry of all open option legs.

    Replaces the single-leg dataclass that silently overwrote on the second
    on_position_opened() call. Now supports concurrent legs (roll overlap,
    multi-expiry strangles, etc).

    Strategy logic queries via:
      is_open        -> any leg exists
      legs           -> dict[expiry_ts, _Leg]
      primary_leg    -> leg closest to target DTE (what roll/exit logic cares about)
    """

    def __init__(self) -> None:
        self.legs: dict[int, _Leg] = {}   # expiry_ts -> _Leg

    @property
    def is_open(self) -> bool:
        return bool(self.legs)

    @property
    def primary_leg(self) -> _Leg | None:
        """Leg with smallest DTE — the one most likely to need rolling."""
        if not self.legs:
            return None
        return min(self.legs.values(), key=lambda l: l.dte_hours())

    # convenience shims for code that used the old single-leg attributes
    @property
    def expiry_ts(self) -> int:
        l = self.primary_leg
        return l.expiry_ts if l else 0

    @property
    def strike(self) -> float:
        l = self.primary_leg
        return l.strike if l else 0.0

    @property
    def iv_at_entry(self) -> float:
        l = self.primary_leg
        return l.iv_at_entry if l else 0.0

    @property
    def notional_usd(self) -> float:
        return sum(l.notional_usd for l in self.legs.values())

    def dte_hours(self) -> float:
        l = self.primary_leg
        return l.dte_hours() if l else 0.0

    def open(self, expiry_ts: int, strike: float, iv: float, notional: float) -> None:
        self.legs[expiry_ts] = _Leg(
            expiry_ts     = expiry_ts,
            strike        = strike,
            iv_at_entry   = iv,
            entry_time_ms = _now_ms(),
            notional_usd  = notional,
        )

    def close(self, expiry_ts: int | None = None) -> None:
        """Close one leg (by expiry) or all legs if expiry_ts is None."""
        if expiry_ts is None:
            self.legs.clear()
        else:
            self.legs.pop(expiry_ts, None)


class GammaScalpStrategy:
    """
    Short gamma, delta-neutral.

    Flow:
      1. Check if state is sane (tradeable, no emergency)
      2. If no position: check entry signal, size it, return ENTER
      3. If position open: check roll/exit/reduce, then check hedge
      4. Return a single StrategySignal per tick

    One signal per call. Execution layer does the actual work.
    This class should have zero I/O, zero async, zero side effects.
    """

    def __init__(self, cfg: Config, state: StateEngine, asset: str) -> None:
        self.cfg   = cfg
        self.state = state
        self.asset = asset

        self.pricer   = ASPricer(cfg)
        self.sizer    = Sizer(cfg, asset)
        self.roller   = RollDetector(cfg)
        self.position = PositionState()

        self._vps     = cfg.strategy.vol_premium_signal
        self._trade_intervals: list[float] = []
        self._last_fill_time:  float       = 0.0
        self._neg_edge_since_ms: int       = 0   # wall clock, not tick accumulator
        self._last_signal_ms:    int       = 0
        self._ofi_entry_threshold: float   = 0.60

        log.info(f"strategy ready | asset={asset} mode={cfg.strategy.strategy.mode}")

    def on_fill(self, fill_time: float | None = None) -> None:
        """Call after any option fill to track arrival rate for k calibration."""
        t = fill_time or time.monotonic()
        if self._last_fill_time > 0:
            self._trade_intervals.append(t - self._last_fill_time)
        self._last_fill_time = t

        as_cfg = self.cfg.strategy.avellaneda_stoikov
        if len(self._trade_intervals) >= as_cfg.calibration_min_trades:
            self.pricer.recalibrate_k(self._trade_intervals[-as_cfg.calibration_min_trades:])

    def on_position_opened(self, expiry_ts: int, strike: float, iv: float, notional: float) -> None:
        self.position.open(expiry_ts, strike, iv, notional)
        log.info(f"position opened | expiry={expiry_ts} strike={strike} iv={iv:.1%} notional=${notional:,.0f} legs={len(self.position.legs)}")

    def on_position_closed(self, expiry_ts: int | None = None) -> None:
        self.position.close(expiry_ts)
        log.info(f"position closed | expiry={expiry_ts} remaining_legs={len(self.position.legs)}")

    def tick(self) -> StrategySignal:
        """
        Main entry point. Call on every meaningful market data update.
        Fast path first - most ticks should hit HOLD or HEDGE and return early.
        """
        try:
            return self._evaluate()
        except StateError as e:
            # state not ready yet, probably warming up
            log.debug(f"strategy waiting for state: {e}")
            return StrategySignal(action=StrategyAction.HOLD, asset=self.asset, reason=f"state not ready: {e}")

    def _evaluate(self) -> StrategySignal:
        state = self.state

        if not state.is_tradeable():
            return self._hold("book not tradeable")

        # emergency check - runs regardless of position state
        if state.signal_emergency():
            self._track_negative_premium(is_emergency=True)
            if self.position.is_open:
                return StrategySignal(
                    action = StrategyAction.FLATTEN,
                    asset  = self.asset,
                    reason = f"emergency: premium={state.vol_premium.premium():.1%}",
                )
            return self._hold("emergency but no position")

        self._track_negative_premium(is_emergency=False)

        if self._neg_edge_hours() >= self.cfg.risk.kill_switch.premium_negative_consecutive_h:
            if self.position.is_open:
                return StrategySignal(
                    action = StrategyAction.REDUCE,
                    asset  = self.asset,
                    target_notional_usd = self.position.notional_usd * 0.30,
                    reason = f"premium negative for {self._neg_edge_hours():.1f}h",
                )

        if self.position.is_open:
            return self._manage_position()
        else:
            return self._check_entry()

    def _check_entry(self) -> StrategySignal:
        state = self.state

        if not state.signal_enter():
            return self._hold("premium below entry threshold")

        # OFI filter: don't enter into strong directional flow.
        # If perp book shows heavy one-sided pressure, the move is likely to
        # continue - our short gamma delta hedge will be chasing.
        # Threshold: |OFI| > 0.6 means 80%+ of recent flow is one-sided.
        ofi = state.ofi()
        if abs(ofi) > self._ofi_entry_threshold:
            return self._hold(f"OFI filter: {ofi:+.2f} (directional flow, skip)")

        spot    = state.spot()
        rv      = state.rv()
        premium = state.vol_premium.premium()
        mult    = state.size_multiplier()

        target = self.sizer.target_notional(
            vol_premium     = premium,
            size_multiplier = mult,
            entry_threshold = self._vps.entry_threshold,
        )

        if target <= 0:
            return self._hold("sizer returned 0")

        if state.vol_surface.n_strikes == 0:
            return self._hold("vol surface empty, waiting for greeks")

        try:
            expiry_ts = self._nearest_expiry()
            iv        = state.atm_iv(expiry_ts)
            dte_h     = _dte_hours(expiry_ts)
            inventory = state.inventory.net_vega
            # SABR vanna for skew-aware spread adjustment
            atm_strike = round(spot / 1000) * 1000
            vanna      = state.sabr_vanna(expiry_ts, atm_strike)
        except StateError as e:
            return self._hold(f"quote inputs missing: {e}")

        quote = self.pricer.quote(
            mid_vol   = iv,
            rv        = rv,
            dte_hours = dte_h,
            inventory = inventory,
            vanna     = vanna,
            spot      = spot,
        )

        regime = state.funding_regime()
        log.info(
            f"ENTER signal | premium={premium:.1%} rv={rv:.1%} iv={iv:.1%} "
            f"notional=${target:,.0f} regime={regime.name} "
            f"spread={quote.spread_vol:.2f}vp ofi={ofi:+.2f} "
            f"vanna={vanna:.4f} sabr={'yes' if state.has_sabr(expiry_ts) else 'no'}"
        )

        return StrategySignal(
            action               = StrategyAction.ENTER,
            asset                = self.asset,
            quote                = quote,
            target_notional_usd  = target,
            reason               = f"premium={premium:.1%} mult={mult:.1f} ofi={ofi:+.2f}",
        )

    def _manage_position(self) -> StrategySignal:
        state   = self.state
        pos     = self.position
        primary = pos.primary_leg

        if primary is None:
            return self._hold("position marked open but no legs found")

        if state.signal_exit():
            return StrategySignal(
                action = StrategyAction.EXIT,
                asset  = self.asset,
                reason = f"premium compressed: {state.vol_premium.premium():.1%}",
            )

        try:
            spot     = state.spot()
            iv_now   = state.atm_iv(primary.expiry_ts)
            dte_h    = primary.dte_hours()
            do_roll, roll_reason = self.roller.should_roll(
                dte_hours   = dte_h,
                spot        = spot,
                strike      = primary.strike,
                iv_now      = iv_now,
                iv_at_entry = primary.iv_at_entry,
            )
        except StateError as e:
            log.warning(f"roll check failed: {e}")
            do_roll, roll_reason = False, ""

        if do_roll:
            log.info(f"ROLL triggered: {roll_reason}")
            return StrategySignal(
                action = StrategyAction.ROLL,
                asset  = self.asset,
                reason = roll_reason,
            )

        hedge = compute_hedge(state, self.asset, self.cfg)
        if hedge is not None:
            return StrategySignal(
                action = StrategyAction.HEDGE,
                asset  = self.asset,
                hedge  = hedge,
                reason = hedge.reason,
            )

        return self._hold("position ok, no action")

    def _track_negative_premium(self, is_emergency: bool) -> None:
        """
        Track how long adjusted edge has been negative.
        Uses wall clock (monotonic ms), not tick accumulation.
        Changing tick_interval_s in main no longer affects this trigger.
        """
        try:
            premium = self.state.vol_premium.premium()
            funding = self.state.funding.rate_ann
            adj     = premium + funding
        except StateError:
            return

        now_ms = _now_ms()
        if adj < 0 or is_emergency:
            if self._neg_edge_since_ms == 0:
                self._neg_edge_since_ms = now_ms
        else:
            self._neg_edge_since_ms = 0

    def _neg_edge_hours(self) -> float:
        """How long adjusted edge has been continuously negative, in hours."""
        if self._neg_edge_since_ms == 0:
            return 0.0
        return (_now_ms() - self._neg_edge_since_ms) / 3_600_000

    def _nearest_expiry(self) -> int:
        """
        Pick the expiry closest to target_dte from the vol surface.
        Hacky but fine - we only have a handful of expiries at any time.
        """
        sc       = self.cfg.strategy
        target_h = sc.leg[self.asset].target_dte_days * 24
        now_ts   = int(time.time())
        expiries = self.state.vol_surface.expiries()

        if not expiries:
            raise StateError("no expiries in vol surface")

        min_h = sc.leg[self.asset].min_dte_days * 24
        max_h = sc.leg[self.asset].max_dte_days * 24
        valid = [
            e for e in expiries
            if min_h <= (e - now_ts) / 3600 <= max_h
        ]

        if not valid:
            raise StateError(f"no expiry in [{min_h}h, {max_h}h] window")

        return min(valid, key=lambda e: abs((e - now_ts) / 3600 - target_h))

    def _hold(self, reason: str) -> StrategySignal:
        return StrategySignal(action=StrategyAction.HOLD, asset=self.asset, reason=reason)

    def status(self) -> dict:
        """Snapshot for logging. Same pattern as StateEngine.snapshot()."""
        pos = self.position
        primary = pos.primary_leg
        try:
            premium = self.state.vol_premium.premium()
            mult    = self.state.size_multiplier()
            rv      = self.state.rv()
        except StateError:
            premium = mult = rv = float("nan")

        return {
            "asset":            self.asset,
            "position_open":    pos.is_open,
            "n_legs":           len(pos.legs),
            "strike":           primary.strike if primary else 0.0,
            "dte_h":            primary.dte_hours() if primary else 0.0,
            "notional_usd":     pos.notional_usd,
            "vol_premium":      premium,
            "rv":               rv,
            "size_mult":        mult,
            "neg_edge_h":       self._neg_edge_hours(),
            "k_mle":            self.pricer.k_current,
            "fills_tracked":    len(self._trade_intervals),
            "ofi":              self.state.ofi(),
            "accumulated_delta": self.state.delta_tracker.accumulated_delta,
        }


# ---- util -------------------------------------------------------------------

def _now_ms() -> int:
    return int(time.monotonic() * 1000)


def _dte_hours(expiry_ts: int) -> float:
    remaining_ms = expiry_ts * 1000 - int(time.time() * 1000)
    return max(0.0, remaining_ms / 3_600_000)

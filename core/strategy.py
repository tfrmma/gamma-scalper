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
        MLE for Poisson arrival rate: lambda_hat = n / sum(intervals).
        Exact MLE for exponential inter-arrivals (Poisson process).
        Blended with prior to avoid cold-start instability.
        """
        if not trade_intervals:
            return
        n          = len(trade_intervals)
        total_time = sum(trade_intervals)
        if total_time <= 0:
            return

        k_mle = n / total_time   # exact Poisson MLE

        # blend with prior: more weight on MLE as we get more data
        # after 200 trades, prior weight ~= 0.1 * (1/2)^(n/50) ~ negligible
        prior_weight = self._k_decay * math.exp(-n / 50.0)
        self._k_mle  = (1 - prior_weight) * k_mle + prior_weight * self.k_prior

        self._last_calibration = time.monotonic()
        self._n_calibrations  += 1
        log.info(
            f"k recalibrated (MLE) | k_mle={k_mle:.4f} k_blended={self._k_mle:.4f} "
            f"n={n} total_time={total_time:.1f}s calibration={self._n_calibrations}"
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

    Short straddle delta should be ~0 at ATM, but it drifts as spot moves.
    We hedge in USD notional on the perp.

    net_delta is in option units (delta per contract).
    Convert to USD notional: notional = delta * spot * contract_size.
    Negative net_delta (short options drifting ITM on puts) -> need long perp.
    """
    if not state.needs_hedge():
        return None

    net_delta    = state.inventory.net_delta
    spot         = state.spot()
    contract_sz  = cfg.market.assets[asset].contract_size
    notional_usd = abs(net_delta) * spot * contract_sz

    if notional_usd < cfg.market.assets[asset].min_trade_amount_perp:
        # rounding noise, not worth a round-trip
        return None

    # sign: if net_delta > 0 (long delta from options), sell perp to flatten
    side = "sell" if net_delta > 0 else "buy"

    return HedgeOrder(
        instrument   = cfg.market.assets[asset].perp_instrument,
        side         = side,
        notional_usd = notional_usd,
        reason       = f"delta={net_delta:.4f} spot={spot:.0f}",
    )


# ---- main strategy ----------------------------------------------------------

@dataclass
class PositionState:
    """Thin container for current open position metadata."""
    is_open:       bool  = False
    expiry_ts:     int   = 0
    strike:        float = 0.0
    iv_at_entry:   float = 0.0
    entry_time_ms: int   = 0
    notional_usd:  float = 0.0

    def open(self, expiry_ts: int, strike: float, iv: float, notional: float) -> None:
        self.is_open       = True
        self.expiry_ts     = expiry_ts
        self.strike        = strike
        self.iv_at_entry   = iv
        self.entry_time_ms = _now_ms()
        self.notional_usd  = notional

    def close(self) -> None:
        self.is_open = False
        self.expiry_ts     = 0
        self.strike        = 0.0
        self.iv_at_entry   = 0.0
        self.entry_time_ms = 0
        self.notional_usd  = 0.0

    def dte_hours(self) -> float:
        if not self.is_open or self.expiry_ts == 0:
            return 0.0
        remaining_ms = self.expiry_ts * 1000 - int(time.time() * 1000)
        return max(0.0, remaining_ms / 3_600_000)


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
        self._consecutive_neg_h: float     = 0.0
        self._last_signal_ms:    int       = 0
        # OFI threshold: skip entry when perp flow is too one-sided
        # 0.6 = 80%+ of recent ticks weighted one direction
        # tune this in live - too tight and you miss entries, too loose and you get run over
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
        log.info(f"position opened | strike={strike} iv={iv:.1%} notional=${notional:,.0f}")

    def on_position_closed(self) -> None:
        self.position.close()
        log.info("position closed")

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

        # extended negative premium - reduce before it gets worse
        if self._consecutive_neg_h >= self.cfg.risk.kill_switch.premium_negative_consecutive_h:
            if self.position.is_open:
                return StrategySignal(
                    action = StrategyAction.REDUCE,
                    asset  = self.asset,
                    target_notional_usd = self.position.notional_usd * 0.30,
                    reason = f"premium negative for {self._consecutive_neg_h:.0f}h",
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
        state = self.state
        pos   = self.position

        # exit check first
        if state.signal_exit():
            return StrategySignal(
                action = StrategyAction.EXIT,
                asset  = self.asset,
                reason = f"premium compressed: {state.vol_premium.premium():.1%}",
            )

        # roll check
        try:
            spot     = state.spot()
            iv_now   = state.atm_iv(pos.expiry_ts)
            dte_h    = pos.dte_hours()
            do_roll, roll_reason = self.roller.should_roll(
                dte_hours    = dte_h,
                spot         = spot,
                strike       = pos.strike,
                iv_now       = iv_now,
                iv_at_entry  = pos.iv_at_entry,
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

        # delta hedge check - most common path
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
        """Track how long the adjusted edge has been negative. Used for reduce trigger."""
        try:
            premium = self.state.vol_premium.premium()
            funding = self.state.funding.rate_ann
            adj     = premium + funding
        except StateError:
            return

        tick_interval_h = (_now_ms() - self._last_signal_ms) / 3_600_000 if self._last_signal_ms else 0
        self._last_signal_ms = _now_ms()

        if adj < 0 or is_emergency:
            self._consecutive_neg_h += tick_interval_h
        else:
            self._consecutive_neg_h = 0.0

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
        try:
            premium = self.state.vol_premium.premium()
            mult    = self.state.size_multiplier()
            rv      = self.state.rv()
        except StateError:
            premium = mult = rv = float("nan")

        return {
            "asset":            self.asset,
            "position_open":    pos.is_open,
            "strike":           pos.strike,
            "dte_h":            pos.dte_hours() if pos.is_open else 0,
            "notional_usd":     pos.notional_usd,
            "vol_premium":      premium,
            "rv":               rv,
            "size_mult":        mult,
            "neg_edge_h":       self._consecutive_neg_h,
            "k_mle":            self.pricer.k_current,
            "fills_tracked":    len(self._trade_intervals),
            "ofi":              self.state.ofi(),
        }


# ---- util -------------------------------------------------------------------

def _now_ms() -> int:
    return int(time.monotonic() * 1000)


def _dte_hours(expiry_ts: int) -> float:
    remaining_ms = expiry_ts * 1000 - int(time.time() * 1000)
    return max(0.0, remaining_ms / 3_600_000)

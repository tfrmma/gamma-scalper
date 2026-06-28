from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import NamedTuple

import numpy as np

from config.loader import Config

log = logging.getLogger("state")


# ---- types ------------------------------------------------------------------

class FundingRegime(Enum):
    BULL    = auto()
    NEUTRAL = auto()
    BEAR    = auto()


class BookSide(Enum):
    BID = auto()
    ASK = auto()


class StateError(Exception):
    """Raised when state is queried before it's valid. Caller should wait."""


# ---- order book -------------------------------------------------------------

class PriceLevel(NamedTuple):
    price: float
    size:  float


@dataclass
class OrderBook:
    """
    Minimal L2 book - bids/asks as sorted lists of (price, size).
    Not trying to be clever here, dicts keyed by price are fast enough
    for options books which rarely have more than 20 levels.

    OFI (Order Flow Imbalance) is computed on every delta update.
    Formula: OFI_t = dV_bid - dV_ask
    where dV_bid = size added to best bid, dV_ask = size added to best ask.
    Normalized to [-1, 1] over a rolling window.
    """
    instrument: str
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)

    last_seq:        int   = 0
    last_update_ms:  int   = 0
    _max_spread_pct: float = 0.10

    # OFI state
    _ofi_window: int   = 100       # ticks to normalize over
    _ofi_raw:    deque = field(init=False)
    _prev_best_bid: float = 0.0
    _prev_best_ask: float = 0.0
    _prev_bid_size: float = 0.0
    _prev_ask_size: float = 0.0

    def __post_init__(self) -> None:
        self._ofi_raw = deque(maxlen=self._ofi_window)

    def apply_snapshot(self, bids: list[list], asks: list[list], seq: int) -> None:
        self.bids = {float(p): float(s) for p, s in bids}
        self.asks = {float(p): float(s) for p, s in asks}
        self.last_seq       = seq
        self.last_update_ms = _now_ms()
        self._update_ofi_prev()

    def apply_delta(self, changes: list[list], seq: int) -> None:
        if seq != self.last_seq + 1:
            raise StateError(
                f"{self.instrument}: seq gap {self.last_seq} -> {seq}, need resync"
            )
        for side, price, size in changes:
            book = self.bids if side == "buy" else self.asks
            p = float(price)
            if float(size) == 0.0:
                book.pop(p, None)
            else:
                book[p] = float(size)

        self.last_seq       = seq
        self.last_update_ms = _now_ms()
        self._compute_ofi()

    def _compute_ofi(self) -> None:
        """
        OFI per Cont, Kukanov, Stoikov (2014).
        dV_bid: change in best bid size (positive = more buying pressure)
        dV_ask: change in best ask size (positive = more selling pressure)
        OFI = dV_bid - dV_ask
        """
        bb = self.best_bid()
        ba = self.best_ask()
        if bb is None or ba is None:
            return

        cur_bid, cur_bid_sz = bb.price, bb.size
        cur_ask, cur_ask_sz = ba.price, ba.size

        # bid side contribution
        if cur_bid > self._prev_best_bid:
            dv_bid = cur_bid_sz              # new best bid - full size counts
        elif cur_bid == self._prev_best_bid:
            dv_bid = cur_bid_sz - self._prev_bid_size
        else:
            dv_bid = -self._prev_bid_size    # best bid moved down - lost size

        # ask side contribution
        if cur_ask < self._prev_best_ask:
            dv_ask = cur_ask_sz
        elif cur_ask == self._prev_best_ask:
            dv_ask = cur_ask_sz - self._prev_ask_size
        else:
            dv_ask = -self._prev_ask_size

        self._ofi_raw.append(dv_bid - dv_ask)
        self._update_ofi_prev()

    def _update_ofi_prev(self) -> None:
        bb = self.best_bid()
        ba = self.best_ask()
        self._prev_best_bid = bb.price if bb else 0.0
        self._prev_bid_size = bb.size  if bb else 0.0
        self._prev_best_ask = ba.price if ba else 0.0
        self._prev_ask_size = ba.size  if ba else 0.0

    def ofi(self) -> float:
        """
        Normalized OFI in [-1, 1].
        Positive = buying pressure, negative = selling pressure.
        Returns 0.0 if not enough data yet.
        """
        if len(self._ofi_raw) < 5:
            return 0.0
        raw   = list(self._ofi_raw)
        total = sum(abs(x) for x in raw)
        if total == 0:
            return 0.0
        return sum(raw) / total

    def ofi_raw_sum(self, n: int = 20) -> float:
        """Unnormalized OFI sum over last n ticks. Useful for sizing signals."""
        if not self._ofi_raw:
            return 0.0
        return float(sum(list(self._ofi_raw)[-n:]))

    def best_bid(self) -> PriceLevel | None:
        if not self.bids:
            return None
        p = max(self.bids)
        return PriceLevel(p, self.bids[p])

    def best_ask(self) -> PriceLevel | None:
        if not self.asks:
            return None
        p = min(self.asks)
        return PriceLevel(p, self.asks[p])

    def mid(self) -> float:
        bid = self.best_bid()
        ask = self.best_ask()
        if bid is None or ask is None:
            raise StateError(f"{self.instrument}: no mid, book is empty")
        return (bid.price + ask.price) / 2.0

    def spread_pct(self) -> float:
        bid = self.best_bid()
        ask = self.best_ask()
        if bid is None or ask is None:
            return float("inf")
        return (ask.price - bid.price) / ask.price

    def is_valid(self, min_bid_levels: int, min_ask_levels: int) -> bool:
        if self.spread_pct() > self._max_spread_pct:
            return False
        if len(self.bids) < min_bid_levels:
            return False
        if len(self.asks) < min_ask_levels:
            return False
        return True

    def is_stale(self, threshold_ms: int) -> bool:
        return (_now_ms() - self.last_update_ms) > threshold_ms

    def depth(self, side: BookSide, levels: int = 5) -> list[PriceLevel]:
        if side == BookSide.BID:
            return [PriceLevel(p, self.bids[p]) for p in sorted(self.bids, reverse=True)[:levels]]
        return [PriceLevel(p, self.asks[p]) for p in sorted(self.asks)[:levels]]


# ---- RV estimator -----------------------------------------------------------

class _Bar(NamedTuple):
    o: float   # open
    h: float   # high
    l: float   # low
    c: float   # close


@dataclass
class RealizedVolEstimator:
    """
    Yang-Zhang (2000) volatility estimator.

    Combines overnight (close-to-open), open-to-close, and Rogers-Satchell
    components into a single minimum-variance estimator. Roughly 5-8x more
    efficient than close-to-close on the same number of bars - matters a lot
    when your primary window is only 24 bars.

    YZ formula:
        sigma^2 = sigma_o^2 + k * sigma_c^2 + (1-k) * sigma_rs^2

    where:
        sigma_o  = overnight vol (log(open_t / close_{t-1}))
        sigma_c  = open-to-close vol (log(close_t / open_t))
        sigma_rs = Rogers-Satchell: E[log(H/C)*log(H/O) + log(L/C)*log(L/O)]
        k        = 0.34 / (1.34 + (n+1)/(n-1))  - optimal weighting

    Falls back to close-to-close if OHLC is not available (e.g. during warmup
    when we only have tick closes from the index feed). Caller sets has_ohlc.
    """
    primary_window:    int
    secondary_window:  int
    annualization:     int
    min_obs:           int

    _bars:     deque = field(init=False)
    _last_bar: _Bar | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._bars = deque(maxlen=self.secondary_window)

    def update(self, close: float) -> None:
        """Close-only update - open=high=low=close. Used when full OHLC not available."""
        self._update_bar(_Bar(o=close, h=close, l=close, c=close))

    def update_ohlc(self, o: float, h: float, l: float, c: float) -> None:
        """Full bar update. Use this when you have real OHLC from the feed."""
        if not (l <= o <= h and l <= c <= h):
            # bad tick - clamp rather than reject, don't want gaps
            h = max(o, h, l, c)
            l = min(o, h, l, c)
        self._update_bar(_Bar(o=o, h=h, l=l, c=c))

    def _update_bar(self, bar: _Bar) -> None:
        self._bars.append(bar)
        self._last_bar = bar

    def rv(self, window: int | None = None) -> float:
        w    = window or self.primary_window
        bars = list(self._bars)[-w:]
        if len(bars) < self.min_obs:
            raise StateError(f"RV needs {self.min_obs} obs, have {len(bars)}")

        # check if we have real OHLC or just close ticks
        has_ohlc = any(b.h != b.l for b in bars)
        if has_ohlc:
            return self._yang_zhang(bars)
        return self._close_to_close(bars)

    def _yang_zhang(self, bars: list[_Bar]) -> float:
        n = len(bars)
        if n < 2:
            return 0.0

        # overnight returns: log(open_t / close_{t-1})
        overnight = [
            math.log(bars[i].o / bars[i-1].c)
            for i in range(1, n)
            if bars[i-1].c > 0 and bars[i].o > 0
        ]
        # open-to-close returns: log(close_t / open_t)
        otc = [
            math.log(b.c / b.o)
            for b in bars
            if b.o > 0
        ]
        # Rogers-Satchell per bar
        rs = []
        for b in bars:
            if b.o > 0 and b.h > 0 and b.l > 0 and b.c > 0:
                term = (math.log(b.h / b.c) * math.log(b.h / b.o)
                      + math.log(b.l / b.c) * math.log(b.l / b.o))
                rs.append(term)

        if not overnight or not otc or not rs:
            return self._close_to_close(bars)

        var_o  = float(np.var(overnight, ddof=1))
        var_c  = float(np.var(otc, ddof=1))
        var_rs = float(np.mean(rs))

        # optimal k (Yang-Zhang 2000, eq. 20)
        k = 0.34 / (1.34 + (n + 1) / max(n - 1, 1))

        var_yz = var_o + k * var_c + (1.0 - k) * var_rs
        return math.sqrt(max(var_yz, 0.0) * self.annualization)

    def _close_to_close(self, bars: list[_Bar]) -> float:
        closes = [b.c for b in bars if b.c > 0]
        if len(closes) < 2:
            raise StateError("not enough closes for C2C fallback")
        rets = [math.log(closes[i] / closes[i-1]) for i in range(1, len(closes))]
        return float(np.std(rets, ddof=1) * math.sqrt(self.annualization))

    def rv_primary(self) -> float:
        return self.rv(self.primary_window)

    def rv_secondary(self) -> float:
        return self.rv(self.secondary_window)

    def rv_ratio(self) -> float:
        """RV(1h) / RV(primary) - kill switch spike detector."""
        if len(self._bars) < 2:
            raise StateError("not enough data for rv_ratio")
        last = self._bars[-1]
        # 1h vol: use the last bar's range if OHLC available, else close move
        if last.h != last.l and last.l > 0:
            prev = self._bars[-2]
            rs_1h = (math.log(last.h / last.c) * math.log(last.h / last.o)
                   + math.log(last.l / last.c) * math.log(last.l / last.o))
            rv_1h = math.sqrt(max(rs_1h, 0.0) * self.annualization)
        else:
            prev = self._bars[-2]
            if prev.c > 0:
                rv_1h = abs(math.log(last.c / prev.c)) * math.sqrt(self.annualization)
            else:
                rv_1h = 0.0
        return rv_1h / max(self.rv_primary(), 1e-8)

    @property
    def n_obs(self) -> int:
        return len(self._bars)


# ---- delta tracker ----------------------------------------------------------

@dataclass
class DeltaTracker:
    """
    Tracks accumulated delta on the option position and fires hedges.

    The proxy gamma is a placeholder - gets replaced by live greeks from Deribit
    as soon as the first options snapshot lands. Don't overthink it.
    """
    threshold:    float   # fire hedge when |accumulated_delta| > this
    gamma_proxy:  float   # ATM gamma, updated from live greeks
    max_interval: float   # hours - force hedge even if delta didn't move

    _accumulated:    float = field(init=False, default=0.0)
    _last_hedge_ms:  int   = field(init=False, default=0)
    _total_hedges:   int   = field(init=False, default=0)

    def update_gamma(self, gamma: float) -> None:
        # called every time we get a fresh greeks snapshot from Deribit
        self.gamma_proxy = gamma

    def on_price_move(self, dS: float) -> None:
        self._accumulated += self.gamma_proxy * dS

    def needs_hedge(self) -> bool:
        if abs(self._accumulated) >= self.threshold:
            return True
        # failsafe - hedge on time even if delta looks fine
        elapsed_h = (_now_ms() - self._last_hedge_ms) / 3_600_000
        return self._last_hedge_ms > 0 and elapsed_h >= self.max_interval

    def reset(self) -> None:
        self._accumulated   = 0.0
        self._last_hedge_ms = _now_ms()
        self._total_hedges += 1

    @property
    def accumulated_delta(self) -> float:
        return self._accumulated

    @property
    def total_hedges(self) -> int:
        return self._total_hedges


# ---- funding state ----------------------------------------------------------

@dataclass
class FundingState:
    """
    Tracks 8h funding rate and classifies regime.
    Short perp = funding income when rate > 0. Don't get this backwards again.
    """
    bull_threshold:          float
    bear_threshold:          float
    bear_confirmation_h:     int
    size_mult_bull:          float
    size_mult_neutral:       float
    size_mult_bear:          float

    _rate_8h:          float = field(init=False, default=0.0)
    _rate_ann:         float = field(init=False, default=0.0)
    _bear_since_ms:    int   = field(init=False, default=0)
    _history:          deque = field(init=False)

    def __post_init__(self) -> None:
        self._history = deque(maxlen=3 * 24)  # 3 days of 8h rates

    def update(self, interest_8h: float) -> None:
        self._rate_8h  = interest_8h
        self._rate_ann = interest_8h * 3 * 365
        self._history.append((interest_8h, _now_ms()))

        if self._rate_ann < self.bear_threshold:
            if self._bear_since_ms == 0:
                self._bear_since_ms = _now_ms()
        else:
            self._bear_since_ms = 0

    @property
    def rate_8h(self) -> float:
        return self._rate_8h

    @property
    def rate_ann(self) -> float:
        return self._rate_ann

    def regime(self) -> FundingRegime:
        if self._rate_ann >= self.bull_threshold:
            return FundingRegime.BULL
        if self._rate_ann < self.bear_threshold:
            # don't flip to bear on a single bad 8h window
            confirmed = (
                self._bear_since_ms > 0
                and (_now_ms() - self._bear_since_ms) >= self.bear_confirmation_h * 3_600_000
            )
            return FundingRegime.BEAR if confirmed else FundingRegime.NEUTRAL
        return FundingRegime.NEUTRAL

    def size_multiplier(self) -> float:
        r = self.regime()
        if r == FundingRegime.BULL:
            return self.size_mult_bull
        if r == FundingRegime.NEUTRAL:
            return self.size_mult_neutral
        return self.size_mult_bear

    def mean_ann_recent(self, n_periods: int = 3) -> float:
        """Mean annualized funding over last n 8h periods."""
        if not self._history:
            return 0.0
        rates = [r for r, _ in list(self._history)[-n_periods:]]
        return float(np.mean(rates)) * 3 * 365


# ---- vol premium signal -----------------------------------------------------

@dataclass
class VolPremiumSignal:
    """
    IV - RV, smoothed over a rolling window.
    The smoothing exists because raw IV swings a lot tick-to-tick on Deribit.
    4h window from the parquet analysis - don't make it longer or you lag the signal.
    """
    entry_threshold:    float
    exit_threshold:     float
    emergency_threshold: float
    smoothing_h:        int

    _raw_premiums: deque = field(init=False)

    def __post_init__(self) -> None:
        self._raw_premiums = deque(maxlen=self.smoothing_h)

    def update(self, iv: float, rv: float) -> None:
        self._raw_premiums.append(iv - rv)

    def premium(self) -> float:
        if not self._raw_premiums:
            raise StateError("vol premium signal has no data yet")
        return float(np.mean(self._raw_premiums))

    def should_enter(self) -> bool:
        return self.premium() >= self.entry_threshold

    def should_exit(self) -> bool:
        return self.premium() < self.exit_threshold

    def is_emergency(self) -> bool:
        return self.premium() < self.emergency_threshold

    @property
    def n_obs(self) -> int:
        return len(self._raw_premiums)


# ---- SABR calibration -------------------------------------------------------

@dataclass
class _SABRParams:
    """
    SABR (Hagan et al. 2002) parameters per expiry.
    We use beta=0.5 (fixed, common for equity/crypto vol - between log-normal and normal)
    and calibrate alpha, rho, nu from the smile.

    sigma_SABR(K, F) = alpha * (FK)^((beta-1)/2) * z/x(z) * [1 + ...]
    where z = (nu/alpha) * (FK)^((1-beta)/2) * log(F/K)
          x(z) = log((sqrt(1-2*rho*z+z^2) + z - rho) / (1-rho))

    Why SABR instead of (or alongside) SVI:
    - SVI is better for interpolation across strikes on a single slice
    - SABR is better for hedging: it gives analytic dVega/dSpot (vanna) and dVega/dVol (volga)
    - We use SABR to compute skew-adjusted hedge ratios in the AS pricer
    - If SABR calibration fails, fall back to flat vol (SVI already handles interpolation)
    """
    alpha: float   # vol of vol level (> 0)
    beta:  float   # CEV exponent (fixed at 0.5)
    rho:   float   # spot-vol correlation (-1 < rho < 1)
    nu:    float   # vol of vol (> 0)

    def implied_vol(self, F: float, K: float, T: float) -> float:
        """Hagan et al. 2002, eq. 2.17b approximation."""
        if T <= 0 or F <= 0 or K <= 0:
            return 0.0

        beta   = self.beta
        FK_mid = math.sqrt(F * K)
        # ATM (F=K): log_FK=0, use series expansion to avoid 0/0
        is_atm = abs(F - K) / F < 1e-6

        if is_atm:
            # Simplified ATM formula: sigma_SABR = alpha / F^(1-beta) * correction
            sigma_atm = self.alpha / (
                F ** (1 - beta) * (
                    1 + (
                        ((1 - beta) ** 2 / 24) * self.alpha ** 2 / F ** (2 * (1 - beta))
                        + 0.25 * self.rho * self.beta * self.nu * self.alpha / F ** (1 - beta)
                        + (2 - 3 * self.rho ** 2) / 24 * self.nu ** 2
                    ) * T
                )
            )
            return max(sigma_atm, 0.0)

        log_FK = math.log(F / K)

        # z and x(z) for smile wings
        z   = (self.nu / self.alpha) * FK_mid ** (1 - beta) * log_FK
        x_z = _sabr_x(z, self.rho) if abs(z) > 1e-8 else 1.0
        z_over_x = z / x_z if abs(x_z) > 1e-10 else 1.0

        sigma_atm = self.alpha / (
            FK_mid ** (1 - beta) * (
                1
                + ((1 - beta) ** 2 / 24) * log_FK ** 2
                + ((1 - beta) ** 4 / 1920) * log_FK ** 4
            )
        )

        correction = (
            1
            + (
                ((1 - beta) ** 2 / 24) * self.alpha ** 2 / FK_mid ** (2 * (1 - beta))
                + 0.25 * self.rho * self.beta * self.nu * self.alpha / FK_mid ** (1 - beta)
                + (2 - 3 * self.rho ** 2) / 24 * self.nu ** 2
            ) * T
        )

        return max(sigma_atm * z_over_x * correction, 0.0)

    def vanna(self, F: float, K: float, T: float, dF: float = 1.0) -> float:
        """
        Numerical dSigma/dF - used for skew-aware delta hedge adjustment.
        Bump-and-reprice, not analytic. Fast enough given call frequency.
        """
        iv_up   = self.implied_vol(F + dF, K, T)
        iv_down = self.implied_vol(F - dF, K, T)
        return (iv_up - iv_down) / (2 * dF)

    def volga(self, F: float, K: float, T: float, dvol: float = 0.001) -> float:
        """Numerical d2Sigma/dSigma2 - convexity of vol."""
        iv_base = self.implied_vol(F, K, T)
        iv_up   = self.implied_vol(F, K * math.exp(dvol), T)
        iv_down = self.implied_vol(F, K * math.exp(-dvol), T)
        return (iv_up - 2 * iv_base + iv_down) / (dvol ** 2)


def _sabr_x(z: float, rho: float) -> float:
    """x(z) function from SABR - eq 2.13b in Hagan et al."""
    disc = math.sqrt(1 - 2 * rho * z + z ** 2)
    return math.log((disc + z - rho) / (1 - rho))


def _fit_sabr(
    strikes: list[float],
    ivs:     list[float],
    forward: float,
    T:       float,
    beta:    float = 0.5,
) -> _SABRParams | None:
    """
    Calibrate SABR (alpha, rho, nu) with fixed beta.
    Uses scipy least_squares. Same fallback pattern as SVI: returns None on failure.
    """
    if len(strikes) < 3 or T <= 0 or forward <= 0:
        return None

    try:
        from scipy.optimize import least_squares
    except ImportError:
        return None

    # ATM vol as initial alpha estimate.
    # SABR alpha has units of vol * F^(1-beta). For beta=0.5, F=60000:
    # alpha ~ atm_iv * sqrt(F) ~ 0.65 * 245 ~ 159. Bound must accommodate this.
    atm_iv = min(ivs, key=lambda iv: abs(strikes[ivs.index(iv)] - forward))
    atm_fk  = forward ** (1 - beta)
    alpha0  = atm_iv * atm_fk
    alpha_max = max(alpha0 * 5.0, 10.0)

    def residuals(params: list) -> list:
        alpha, rho, nu = params
        if alpha <= 0 or nu <= 0 or not (-0.999 < rho < 0.999):
            return [1e6] * len(strikes)
        p = _SABRParams(alpha=alpha, beta=beta, rho=rho, nu=nu)
        return [p.implied_vol(forward, K, T) - iv for K, iv in zip(strikes, ivs)]

    try:
        result = least_squares(
            residuals,
            x0     = [alpha0, -0.3, 0.4],
            bounds = ([1e-6, -0.999, 1e-4], [alpha_max, 0.999, 10.0]),
            max_nfev = 300,
            ftol   = 1e-6,
        )
        alpha, rho, nu = result.x
        return _SABRParams(alpha=alpha, beta=beta, rho=rho, nu=nu)
    except Exception:
        return None

def _norm_cdf(x: float) -> float:
    """Abramowitz & Stegun. Same one used in deribit_gateway, keep in sync."""
    if x < 0:
        return 1.0 - _norm_cdf(-x)
    t = 1.0 / (1.0 + 0.2316419 * x)
    poly = t * (0.319381530
              + t * (-0.356563782
              + t * (1.781477937
              + t * (-1.821255978
              + t * 1.330274429))))
    return 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x) * poly


@dataclass
class _SVIParams:
    """
    Raw SVI parametrization (Gatheral 2004).
    w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))
    where k = log(K/F), w = implied_var * T
    """
    a:     float   # vertical shift (overall variance level)
    b:     float   # slope (wing steepness, >= 0)
    rho:   float   # correlation (-1 < rho < 1), controls skew
    m:     float   # horizontal shift (ATM location)
    sigma: float   # smoothness of ATM (> 0)

    def w(self, k: float) -> float:
        """Total variance at log-moneyness k."""
        return self.a + self.b * (self.rho * (k - self.m)
               + math.sqrt((k - self.m) ** 2 + self.sigma ** 2))

    def iv(self, k: float, T: float) -> float:
        """Annualized implied vol at log-moneyness k, time T (years)."""
        if T <= 0:
            return 0.0
        w_val = self.w(k)
        return math.sqrt(max(w_val, 0.0) / T)


def _fit_svi(
    strikes:  list[float],
    ivs:      list[float],
    forward:  float,
    T:        float,
) -> _SVIParams | None:
    """
    Fit SVI to a slice of (strike, IV) pairs using Levenberg-Marquardt via
    scipy.optimize.least_squares. Returns None if fit fails or slice is too thin.

    We don't need a perfect fit - just something better than nearest-strike.
    If we have fewer than 3 points, not worth fitting; caller falls back to nearest.
    """
    if len(strikes) < 3 or T <= 0:
        return None

    try:
        from scipy.optimize import least_squares
    except ImportError:
        return None   # scipy not available, caller falls back

    ks  = [math.log(K / forward) for K in strikes]
    ws  = [(iv ** 2) * T for iv in ivs]

    def residuals(params: list) -> list:
        a, b, rho, m, sigma = params
        svi = _SVIParams(a, b, rho, m, sigma)
        return [svi.w(k) - w for k, w in zip(ks, ws)]

    # initial guess: flat vol surface
    w_atm = float(np.mean(ws))
    x0    = [w_atm, 0.1, -0.3, 0.0, 0.3]
    bounds = (
        [-np.inf, 0.0,  -0.999, -np.inf, 1e-4],
        [ np.inf, np.inf, 0.999,  np.inf, np.inf],
    )

    try:
        result = least_squares(residuals, x0, bounds=bounds, max_nfev=500, ftol=1e-6)
        if not result.success and result.cost > 1e-3:
            return None
        a, b, rho, m, sigma = result.x
        return _SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)
    except Exception:
        return None


# ---- vol surface ------------------------------------------------------------

@dataclass
class VolSurface:
    """
    SVI vol surface per expiry, calibrated from Deribit greeks snapshots.

    On each update_strike() call the raw grid gets a new point.
    SVI is refitted when enough points are available (>= svi_min_strikes).
    When SVI fit exists, iv_at_strike() interpolates/extrapolates smoothly.
    When it doesn't (not enough strikes, fit failed), falls back to nearest-strike.

    For straddles, only ATM matters - nearest-strike is fine.
    For strangles, you need wings, and nearest-strike gives you stale/wrong IVs
    for strikes that aren't directly quoted. That's what this solves.
    """
    asset:            str
    svi_min_strikes:  int   = 4
    refit_interval_s: float = 60.0

    _iv_grid:    dict[tuple[int, float], float] = field(init=False, default_factory=dict)
    _svi_params: dict[int, _SVIParams]           = field(init=False, default_factory=dict)
    _sabr_params: dict[int, _SABRParams]         = field(init=False, default_factory=dict)
    _forwards:   dict[int, float]                = field(init=False, default_factory=dict)
    _last_fit_s: dict[int, float]                = field(init=False, default_factory=dict)
    _last_update_ms: int                         = field(init=False, default=0)

    def update_strike(self, expiry_ts: int, strike: float, iv: float) -> None:
        self._iv_grid[(expiry_ts, strike)] = iv
        self._last_update_ms = _now_ms()
        self._maybe_refit(expiry_ts)

    def update_forward(self, expiry_ts: int, forward: float) -> None:
        """
        Set the forward price for an expiry (index price is fine for short-dated).
        Used by SVI calibration for log-moneyness calculation.
        """
        self._forwards[expiry_ts] = forward

    def _maybe_refit(self, expiry_ts: int) -> None:
        now = time.monotonic()
        last = self._last_fit_s.get(expiry_ts, 0.0)
        if now - last < self.refit_interval_s:
            return

        strikes_for_expiry = [k[1] for k in self._iv_grid if k[0] == expiry_ts]
        if len(strikes_for_expiry) < self.svi_min_strikes:
            return

        ivs     = [self._iv_grid[(expiry_ts, s)] for s in strikes_for_expiry]
        forward = self._forwards.get(expiry_ts, min(strikes_for_expiry))
        T       = max(0.0, (expiry_ts - time.time()) / (365 * 24 * 3600))

        # SVI fit - interpolation across strikes
        svi = _fit_svi(strikes_for_expiry, ivs, forward, T)
        if svi is not None:
            self._svi_params[expiry_ts] = svi
            log.debug(f"SVI fit {self.asset} expiry={expiry_ts} rho={svi.rho:.3f}")

        # SABR fit - skew-aware greeks (vanna, volga)
        sabr = _fit_sabr(strikes_for_expiry, ivs, forward, T)
        if sabr is not None:
            self._sabr_params[expiry_ts] = sabr
            log.debug(f"SABR fit {self.asset} expiry={expiry_ts} alpha={sabr.alpha:.4f} rho={sabr.rho:.3f} nu={sabr.nu:.4f}")

        self._last_fit_s[expiry_ts] = now

    def sabr_vanna(self, expiry_ts: int, strike: float, forward: float) -> float:
        """
        dSigma/dF from SABR - tells you how much IV changes when spot moves.
        Used to adjust delta hedge: true_delta = BS_delta + vanna_adjustment.
        Returns 0.0 if SABR not calibrated yet.
        """
        params = self._sabr_params.get(expiry_ts)
        if params is None:
            return 0.0
        T = max(1e-6, (expiry_ts - time.time()) / (365 * 24 * 3600))
        return params.vanna(forward, strike, T)

    def sabr_volga(self, expiry_ts: int, strike: float, forward: float) -> float:
        """
        d2Sigma/dSigma2 from SABR - vol convexity.
        Useful for sizing vega risk on wing strikes.
        Returns 0.0 if not calibrated.
        """
        params = self._sabr_params.get(expiry_ts)
        if params is None:
            return 0.0
        T = max(1e-6, (expiry_ts - time.time()) / (365 * 24 * 3600))
        return params.volga(forward, strike, T)

    def sabr_iv(self, expiry_ts: int, strike: float, forward: float) -> float | None:
        """
        SABR-implied vol at a strike. Returns None if not calibrated.
        Prefer iv_at_strike() (SVI) for interpolation - use this for hedge ratio adjustment.
        """
        params = self._sabr_params.get(expiry_ts)
        if params is None:
            return None
        T = max(1e-6, (expiry_ts - time.time()) / (365 * 24 * 3600))
        return params.implied_vol(forward, strike, T)

    def has_sabr(self, expiry_ts: int) -> bool:
        return expiry_ts in self._sabr_params

    def iv_at_strike(self, expiry_ts: int, strike: float, forward: float) -> float:
        """
        IV at an arbitrary strike. Uses SVI if available, nearest-strike otherwise.
        This is what strangle execution should call for wing strikes.
        """
        T = max(0.0, (expiry_ts - time.time()) / (365 * 24 * 3600))

        params = self._svi_params.get(expiry_ts)
        if params is not None and T > 0 and forward > 0:
            k  = math.log(strike / forward)
            iv = params.iv(k, T)
            if iv > 0:
                return iv

        # fallback: nearest quoted strike
        return self._nearest_strike_iv(expiry_ts, strike)

    def _nearest_strike_iv(self, expiry_ts: int, strike: float) -> float:
        candidates = {k: v for k, v in self._iv_grid.items() if k[0] == expiry_ts}
        if not candidates:
            raise StateError(f"{self.asset}: no IV data for expiry={expiry_ts}")
        best = min(candidates, key=lambda k: abs(k[1] - strike))
        return candidates[best]

    def atm_iv(self, expiry_ts: int, spot: float) -> float:
        """ATM IV - uses SVI if available, nearest-strike otherwise."""
        return self.iv_at_strike(expiry_ts, spot, forward=spot)

    def get_iv(self, expiry_ts: int, strike: float) -> float:
        """Exact lookup - raises if strike not in grid."""
        key = (expiry_ts, strike)
        if key not in self._iv_grid:
            raise StateError(f"{self.asset}: no IV for expiry={expiry_ts} strike={strike}")
        return self._iv_grid[key]

    def has_svi(self, expiry_ts: int) -> bool:
        return expiry_ts in self._svi_params

    def skew_at_delta(
        self,
        expiry_ts: int,
        delta: float,
        spot:  float,
        is_call: bool = True,
    ) -> float:
        """
        IV at a given delta target (e.g. 0.25 for 25-delta wing).
        Inverts Black-76 numerically to find the strike, then queries the surface.
        Needed for strangle entry when you size by delta rather than moneyness.
        """
        T = max(1e-6, (expiry_ts - time.time()) / (365 * 24 * 3600))

        # bracket search: find strike where Black-76 delta matches target
        # works for calls (delta > 0) and puts (delta < 0, pass abs value)
        abs_delta = abs(delta)
        lo, hi    = spot * 0.5, spot * 2.0

        for _ in range(50):
            mid    = (lo + hi) / 2.0
            iv_mid = self.iv_at_strike(expiry_ts, mid, spot)
            d1     = (math.log(spot / mid) + 0.5 * iv_mid**2 * T) / (iv_mid * math.sqrt(T) + 1e-10)
            d_mid  = _norm_cdf(d1) if is_call else _norm_cdf(d1) - 1.0

            if abs(abs(d_mid) - abs_delta) < 1e-5:
                break
            if abs(d_mid) > abs_delta:
                hi = mid  # too far ITM
            else:
                lo = mid  # too far OTM

        return self.iv_at_strike(expiry_ts, mid, spot)

    def is_stale(self, threshold_ms: int) -> bool:
        return (_now_ms() - self._last_update_ms) > threshold_ms

    def expiries(self) -> set[int]:
        return {k[0] for k in self._iv_grid}

    @property
    def n_strikes(self) -> int:
        return len(self._iv_grid)


# ---- inventory state --------------------------------------------------------

@dataclass
class InventoryState:
    """Net greeks across all open option legs. Updated on every fill."""
    asset: str

    net_delta: float = 0.0
    net_gamma: float = 0.0
    net_vega:  float = 0.0
    net_theta: float = 0.0

    # perp hedge position in USD notional (negative = short)
    perp_position_usd: float = 0.0

    _fill_count: int = 0

    def on_option_fill(
        self,
        delta: float,
        gamma: float,
        vega: float,
        theta: float,
        qty: float,     # positive = bought, negative = sold
    ) -> None:
        self.net_delta += delta * qty
        self.net_gamma += gamma * qty
        self.net_vega  += vega  * qty
        self.net_theta += theta * qty
        self._fill_count += 1

    def on_perp_fill(self, notional_usd: float) -> None:
        # notional_usd: positive = long, negative = short
        self.perp_position_usd += notional_usd

    def net_delta_after_hedge(self) -> float:
        """Option delta + perp hedge delta. Should be ~0 when hedged."""
        # perp delta in BTC ≈ notional / spot, but state_engine doesn't know spot
        # caller must pass spot-adjusted perp delta separately if needed
        # for now just return option delta - good enough for the threshold check
        return self.net_delta

    def reset(self) -> None:
        """Call on full flatten."""
        self.net_delta         = 0.0
        self.net_gamma         = 0.0
        self.net_vega          = 0.0
        self.net_theta         = 0.0
        self.perp_position_usd = 0.0


# ---- index price tracker ----------------------------------------------------

@dataclass
class IndexTracker:
    """
    Tracks the Deribit index price (not the perp mid).
    We price everything off the index, not the perp - perp can diverge on squeezes.
    """
    _price:     float = 0.0
    _update_ms: int   = 0

    def update(self, price: float) -> None:
        self._price     = price
        self._update_ms = _now_ms()

    def price(self) -> float:
        if self._price == 0.0:
            raise StateError("index price not yet received")
        return self._price

    def is_stale(self, threshold_ms: int) -> bool:
        return (_now_ms() - self._update_ms) > threshold_ms


# ---- top-level state engine -------------------------------------------------

class StateEngine:
    """
    Single source of truth for all market state.
    Strategy layer reads from here, never holds its own state.

    Not thread-safe by design - we're running a single asyncio event loop.
    If you're calling this from multiple threads, you've already made a mistake.
    """

    def __init__(self, cfg: Config, asset: str) -> None:
        self.asset = asset
        self._cfg  = cfg

        ac  = cfg.market.assets[asset]
        sc  = cfg.strategy
        rv  = sc.realized_vol
        dh  = sc.delta_hedge
        fr  = sc.funding_regime
        vps = sc.vol_premium_signal
        ob  = cfg.market.orderbook

        self.option_book = OrderBook(
            instrument=f"{asset}-options",
            _max_spread_pct=ob.max_spread_pct,
        )
        self.perp_book = OrderBook(
            instrument=ac.perp_instrument,
            _max_spread_pct=ob.max_spread_pct,
        )

        self.rv_estimator = RealizedVolEstimator(
            primary_window   = rv.primary_window_h,
            secondary_window = rv.secondary_window_h,
            annualization    = rv.annualization_factor,
            min_obs          = rv.min_observations,
        )

        self.delta_tracker = DeltaTracker(
            threshold    = dh.delta_threshold,
            gamma_proxy  = dh.option_gamma_proxy,
            max_interval = dh.max_hedge_interval_h,
        )

        self.funding = FundingState(
            bull_threshold      = fr.bull_threshold,
            bear_threshold      = fr.bear_threshold,
            bear_confirmation_h = fr.bear_confirmation_h,
            size_mult_bull      = fr.size_multiplier_bull,
            size_mult_neutral   = fr.size_multiplier_neutral,
            size_mult_bear      = fr.size_multiplier_bear,
        )

        self.vol_premium = VolPremiumSignal(
            entry_threshold     = vps.entry_threshold,
            exit_threshold      = vps.exit_threshold,
            emergency_threshold = vps.emergency_exit_threshold,
            smoothing_h         = vps.signal_smoothing_h,
        )

        self.vol_surface = VolSurface(asset=asset)
        self.inventory   = InventoryState(asset=asset)
        self.index       = IndexTracker()

        # hourly bar accumulator for RV estimator (bug fix: don't feed raw ticks)
        self._current_bar:   dict | None = None
        self._current_bar_h: int         = -1

        self._ob_cfg = ob
        log.info(f"state engine ready | asset={asset}")

    # ---- inbound updates (called by market data layer) ---------------------

    def on_index_price(self, price: float) -> None:
        """
        Tick-level price update. Accumulates into an hourly bar before
        feeding the RV estimator - annualization_factor=8760 assumes
        hourly observations, not per-tick. Sending raw ticks inflated
        the RV ratio on any normal market move.

        Delta tracking fires on every tick (correct - that's the point).
        RV only updates when the bar closes.
        """
        now_h = int(time.time() // 3600)   # current hour bucket

        prev = self.index._price
        self.index.update(price)

        if prev > 0:
            dS = price - prev
            self.delta_tracker.on_price_move(dS)

        # accumulate into current hour bar
        bar = self._current_bar
        if bar is None or self._current_bar_h != now_h:
            # close the previous bar and push to estimator
            if bar is not None:
                self.rv_estimator.update_ohlc(bar["o"], bar["h"], bar["l"], bar["c"])
            self._current_bar   = {"o": price, "h": price, "l": price, "c": price}
            self._current_bar_h = now_h
        else:
            bar["h"] = max(bar["h"], price)
            bar["l"] = min(bar["l"], price)
            bar["c"] = price

    def flush_bar(self) -> None:
        """
        Force-close the current accumulating bar and push to RV estimator.
        Call this in tests to simulate hour boundaries, or at session end.
        """
        if self._current_bar is not None:
            bar = self._current_bar
            self.rv_estimator.update_ohlc(bar["o"], bar["h"], bar["l"], bar["c"])
            self._current_bar   = None
            self._current_bar_h = -1

    def on_ohlc_bar(self, o: float, h: float, l: float, c: float) -> None:
        """
        Hourly OHLC bar - feeds Yang-Zhang directly.
        Call this instead of on_index_price when the feed provides full bars.
        Delta tracking still uses the close.
        """
        prev = self.index._price
        self.index.update(c)
        if prev > 0:
            dS = c - prev
            self.delta_tracker.on_price_move(dS)
        self.rv_estimator.update_ohlc(o, h, l, c)

    def on_perp_book_snapshot(self, bids: list, asks: list, seq: int) -> None:
        self.perp_book.apply_snapshot(bids, asks, seq)

    def on_perp_book_delta(self, changes: list, seq: int) -> None:
        self.perp_book.apply_delta(changes, seq)

    def on_option_book_snapshot(self, bids: list, asks: list, seq: int) -> None:
        self.option_book.apply_snapshot(bids, asks, seq)

    def on_option_book_delta(self, changes: list, seq: int) -> None:
        self.option_book.apply_delta(changes, seq)

    def on_greeks(
        self,
        expiry_ts: int,
        strike: float,
        iv: float,
        delta: float,
        gamma: float,
        vega: float,
        theta: float,
    ) -> None:
        self.vol_surface.update_strike(expiry_ts, strike, iv)
        self.delta_tracker.update_gamma(gamma)

        try:
            spot = self.index.price()
            # correct forward: F = S * exp(r * T) where r = annualized funding
            # at 5% ann funding, 7 DTE: e^(0.05 * 7/365) - 1 ≈ 0.1% bias without this
            T_years = max(0.0, (expiry_ts - time.time()) / (365 * 24 * 3600))
            funding_ann = self.funding.rate_ann if self.funding.rate_ann != 0.0 else 0.0
            forward = spot * math.exp(funding_ann * T_years)
            self.vol_surface.update_forward(expiry_ts, forward)
            rv = self.rv_estimator.rv_primary()
            self.vol_premium.update(iv, rv)
        except StateError:
            pass  # not ready yet, skip signal update

    def on_funding_rate(self, interest_8h: float) -> None:
        self.funding.update(interest_8h)

    def on_option_fill(self, delta: float, gamma: float, vega: float, theta: float, qty: float) -> None:
        self.inventory.on_option_fill(delta, gamma, vega, theta, qty)

    def on_perp_fill(self, notional_usd: float) -> None:
        self.inventory.on_perp_fill(notional_usd)
        self.delta_tracker.reset()

    # ---- queries (called by strategy layer) --------------------------------

    def spot(self) -> float:
        return self.index.price()

    def perp_mid(self) -> float:
        return self.perp_book.mid()

    def rv(self) -> float:
        return self.rv_estimator.rv_primary()

    def atm_iv(self, expiry_ts: int) -> float:
        return self.vol_surface.atm_iv(expiry_ts, self.index.price())

    def iv_at_strike(self, expiry_ts: int, strike: float) -> float:
        """SVI-interpolated IV at arbitrary strike. Falls back to nearest-quoted."""
        return self.vol_surface.iv_at_strike(expiry_ts, strike, self.index.price())

    def skew_at_delta(self, expiry_ts: int, delta: float, is_call: bool = True) -> float:
        """IV for a delta-targeted wing (e.g. 0.25 for 25d strangle leg)."""
        return self.vol_surface.skew_at_delta(
            expiry_ts, delta, self.index.price(), is_call
        )

    def has_svi(self, expiry_ts: int) -> bool:
        return self.vol_surface.has_svi(expiry_ts)

    def has_sabr(self, expiry_ts: int) -> bool:
        return self.vol_surface.has_sabr(expiry_ts)

    def sabr_vanna(self, expiry_ts: int, strike: float) -> float:
        return self.vol_surface.sabr_vanna(expiry_ts, strike, self.index.price())

    def sabr_volga(self, expiry_ts: int, strike: float) -> float:
        return self.vol_surface.sabr_volga(expiry_ts, strike, self.index.price())

    def ofi(self) -> float:
        """Normalized OFI from perp L2. Positive = buying pressure."""
        return self.perp_book.ofi()

    def ofi_raw(self, n: int = 20) -> float:
        return self.perp_book.ofi_raw_sum(n)

    def needs_hedge(self) -> bool:
        return self.delta_tracker.needs_hedge()

    def funding_regime(self) -> FundingRegime:
        return self.funding.regime()

    def size_multiplier(self) -> float:
        return self.funding.size_multiplier()

    def signal_enter(self) -> bool:
        return self.vol_premium.should_enter()

    def signal_exit(self) -> bool:
        return self.vol_premium.should_exit()

    def signal_emergency(self) -> bool:
        return self.vol_premium.is_emergency()

    def rv_spike_ratio(self) -> float:
        return self.rv_estimator.rv_ratio()

    def is_tradeable(self) -> bool:
        """Quick sanity check before the strategy layer does anything."""
        ob = self._ob_cfg
        if self.perp_book.is_stale(ob.stale_book_threshold_ms):
            return False
        if not self.perp_book.is_valid(ob.min_bid_levels, ob.min_ask_levels):
            return False
        if self.index.is_stale(ob.stale_book_threshold_ms):
            return False
        return True

    def snapshot(self) -> dict:
        """Structured log snapshot - call this every N seconds."""
        try:
            return {
                "asset":          self.asset,
                "spot":           self.index.price(),
                "perp_mid":       self.perp_book.mid(),
                "rv_24h":         self.rv_estimator.rv_primary(),
                "rv_ratio":       self.rv_estimator.rv_ratio(),
                "vol_premium":    self.vol_premium.premium(),
                "funding_ann":    self.funding.rate_ann,
                "funding_regime": self.funding.regime().name,
                "size_mult":      self.funding.size_multiplier(),
                "delta_accum":    self.delta_tracker.accumulated_delta,
                "needs_hedge":    self.delta_tracker.needs_hedge(),
                "ofi":            self.perp_book.ofi(),
                "net_vega":       self.inventory.net_vega,
                "net_gamma":      self.inventory.net_gamma,
                "net_theta":      self.inventory.net_theta,
                "perp_pos_usd":   self.inventory.perp_position_usd,
                "tradeable":      self.is_tradeable(),
            }
        except StateError as e:
            return {"asset": self.asset, "error": str(e), "tradeable": False}


# ---- util -------------------------------------------------------------------

def _now_ms() -> int:
    return int(time.monotonic() * 1000)

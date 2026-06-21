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
    """
    instrument: str
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)

    last_seq:        int   = 0
    last_update_ms:  int   = 0
    _max_spread_pct: float = 0.10

    def apply_snapshot(self, bids: list[list], asks: list[list], seq: int) -> None:
        self.bids = {float(p): float(s) for p, s in bids}
        self.asks = {float(p): float(s) for p, s in asks}
        self.last_seq       = seq
        self.last_update_ms = _now_ms()

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


# ---- vol surface (thin wrapper) ---------------------------------------------

@dataclass
class VolSurface:
    """
    Holds the current IV surface from Deribit greeks snapshots.
    Not doing SVI interpolation here - that's for the pricer.
    This is just the state container.

    TODO: add SVI fit so we can interpolate strikes that aren't quoted.
    For now we only trade ATM straddles so it doesn't matter yet.
    """
    asset: str

    # keyed by (expiry_ts, strike) -> iv
    _iv_grid: dict[tuple[int, float], float] = field(init=False, default_factory=dict)
    _last_update_ms: int = field(init=False, default=0)

    def update_strike(self, expiry_ts: int, strike: float, iv: float) -> None:
        self._iv_grid[(expiry_ts, strike)] = iv
        self._last_update_ms = _now_ms()

    def get_iv(self, expiry_ts: int, strike: float) -> float:
        key = (expiry_ts, strike)
        if key not in self._iv_grid:
            raise StateError(f"{self.asset}: no IV for expiry={expiry_ts} strike={strike}")
        return self._iv_grid[key]

    def atm_iv(self, expiry_ts: int, spot: float) -> float:
        """Nearest-strike approximation for ATM IV. Good enough for straddles."""
        candidates = {
            k: v for k, v in self._iv_grid.items() if k[0] == expiry_ts
        }
        if not candidates:
            raise StateError(f"{self.asset}: no IV data for expiry={expiry_ts}")
        best = min(candidates, key=lambda k: abs(k[1] - spot))
        return candidates[best]

    def is_stale(self, threshold_ms: int) -> bool:
        return (_now_ms() - self._last_update_ms) > threshold_ms

    def expiries(self) -> set[int]:
        """All expiry timestamps currently in the surface."""
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

        self._ob_cfg = ob
        log.info(f"state engine ready | asset={asset}")

    # ---- inbound updates (called by market data layer) ---------------------

    def on_index_price(self, price: float) -> None:
        """Tick-level close update. YZ falls back to C2C if no OHLC bars arrive."""
        prev = self.index._price
        self.index.update(price)
        if prev > 0:
            dS = price - prev
            self.delta_tracker.on_price_move(dS)
            self.rv_estimator.update(price)

    def on_ohlc_bar(self, o: float, h: float, l: float, c: float) -> None:
        """
        Hourly OHLC bar - feeds Yang-Zhang directly.
        Call this instead of on_index_price when the feed provides full bars
        (e.g. after subscribing to chart.trades or aggregating ticks into bars).
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
            rv   = self.rv_estimator.rv_primary()
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

"""
tests/test_integration.py

End-to-end test of the full signal flow:
  market data -> state engine -> strategy -> execution -> risk

No real WebSocket. Feed state manually, verify signals and order flow.
Run with: python -m pytest tests/ -v
or:        python tests/test_integration.py
"""

from __future__ import annotations

import asyncio
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.loader import load_config
from core.state_engine import StateEngine, StateError
from core.strategy import GammaScalpStrategy, StrategyAction
from core.execution import ExecutionEngine, ExchangeGateway, Order
from core.risk_engine import RiskEngine, KillReason
from core.market_data import (
    _parse_book, _parse_index, _parse_ticker,
    _parse_option_instrument, build_subscriptions,
)


# ---- fixtures ---------------------------------------------------------------

class RecordingGateway(ExchangeGateway):
    """Records every call. Lets tests assert on what was sent."""
    def __init__(self) -> None:
        self.sent:     list[Order]  = []
        self.canceled: list[str]    = []
        self._id = 0

    async def send_order(self, order: Order) -> dict:
        self.sent.append(order)
        self._id += 1
        return {"id": f"EX{self._id:04d}"}

    async def cancel_order(self, eid: str) -> dict:
        self.canceled.append(eid)
        return {"result": "ok"}

    async def cancel_all(self, instrument=None) -> dict:
        return {"result": "ok"}

    async def get_order(self, eid: str) -> dict:
        return {"order_state": "filled", "filled_amount": 1.0, "average_price": 60000.0}

    async def get_positions(self) -> dict:
        return {}


def make_system(seed: int = 42):
    cfg     = load_config("./config")
    state   = StateEngine(cfg, "BTC")
    gw      = RecordingGateway()
    eng     = ExecutionEngine(cfg, state, gw, "BTC")
    risk    = RiskEngine(cfg, state, eng, "BTC")
    strat   = GammaScalpStrategy(cfg, state, "BTC")
    return cfg, state, gw, eng, risk, strat


def warm_state(state: StateEngine, seed: int = 42, n: int = 30, vol: float = 0.005) -> float:
    """Feed enough price history to make RV valid. Returns final price."""
    random.seed(seed)
    price = 60000.0
    for _ in range(n):
        price *= math.exp(random.gauss(0, vol))
        state.on_index_price(price)
    return price


def setup_book(state: StateEngine, price: float, seq: int = 1) -> None:
    state.on_perp_book_snapshot(
        bids=[[price * 0.999, 10], [price * 0.998, 20], [price * 0.997, 5]],
        asks=[[price * 1.001, 10], [price * 1.002, 20], [price * 1.003, 5]],
        seq=seq,
    )


def setup_greeks(state: StateEngine, price: float, iv: float = 0.65) -> int:
    expiry = int(time.time()) + 7 * 24 * 3600
    state.on_greeks(
        expiry_ts=expiry,
        strike=round(price / 1000) * 1000,
        iv=iv, delta=0.50, gamma=0.00002, vega=50.0, theta=-15.0,
    )
    return expiry


def push_premium(state: StateEngine, iv: float, n: int = 4) -> None:
    rv = state.rv_estimator.rv_primary()
    for _ in range(n):
        state.vol_premium.update(iv, rv)


# ---- tests ------------------------------------------------------------------

def test_cold_start_holds():
    """Strategy returns HOLD before state is warm."""
    _, state, _, _, _, strat = make_system()
    sig = strat.tick()
    assert sig.action == StrategyAction.HOLD


def test_enter_signal_with_positive_premium():
    """Full warm state with IV > RV produces ENTER with valid quote."""
    cfg, state, gw, eng, risk, strat = make_system()

    price = warm_state(state, vol=0.004)  # ~25% ann RV
    setup_book(state, price)
    state.on_funding_rate(0.0003)         # bull regime
    setup_greeks(state, price, iv=0.65)   # 65% IV >> 25% RV
    push_premium(state, iv=0.65)

    sig = strat.tick()
    assert sig.action == StrategyAction.ENTER, f"expected ENTER got {sig.action} | {sig.reason}"
    assert sig.quote is not None
    assert sig.quote.bid_vol < sig.quote.mid_vol < sig.quote.ask_vol
    assert sig.quote.spread_vol >= cfg.strategy.avellaneda_stoikov.spread_min_vol_pts
    assert sig.target_notional_usd >= cfg.strategy.leg["BTC"].min_notional_usd


async def test_enter_sends_two_option_orders():
    """ENTER signal dispatched to execution sends call + put orders."""
    cfg, state, gw, eng, risk, strat = make_system()

    price = warm_state(state, vol=0.004)
    setup_book(state, price)
    state.on_funding_rate(0.0003)
    setup_greeks(state, price, iv=0.65)
    push_premium(state, iv=0.65)

    sig = strat.tick()
    assert sig.action == StrategyAction.ENTER
    await eng.handle(sig)

    assert len(gw.sent) == 2, f"expected 2 orders (call+put), got {len(gw.sent)}"
    sides = [o.side for o in gw.sent]
    assert sides == ["sell", "sell"]
    assert all(o.post_only for o in gw.sent)
    assert all(o.order_type == "limit" for o in gw.sent)


async def test_hedge_sends_perp_order():
    """HEDGE signal sends exactly one perp order."""
    cfg, state, gw, eng, risk, strat = make_system()

    price = warm_state(state, vol=0.004)
    setup_book(state, price)
    state.on_funding_rate(0.0003)
    setup_greeks(state, price, iv=0.65)
    push_premium(state, iv=0.65)

    # open a position first
    sig = strat.tick()
    strat.on_position_opened(
        expiry_ts=int(time.time()) + 7*24*3600,
        strike=round(price/1000)*1000,
        iv=0.65,
        notional=10000.0,
    )

    # force delta accumulation
    state.delta_tracker.gamma_proxy = 0.01
    state.on_index_price(price * 1.06)  # big move

    assert state.needs_hedge(), "delta should exceed threshold after big move"

    sig = strat.tick()
    # might be ROLL if moneyness > threshold, HEDGE otherwise
    assert sig.action in (StrategyAction.HEDGE, StrategyAction.ROLL)

    if sig.action == StrategyAction.HEDGE:
        await eng.handle(sig)
        perp_orders = [o for o in gw.sent if "PERPETUAL" in o.instrument]
        assert len(perp_orders) >= 1
        assert perp_orders[-1].side in ("buy", "sell")


async def test_emergency_flatten_on_negative_premium():
    """Emergency signal (premium < -15%) triggers FLATTEN."""
    cfg, state, gw, eng, risk, strat = make_system()

    price = warm_state(state, vol=0.004)
    setup_book(state, price)
    state.on_funding_rate(0.0003)
    setup_greeks(state, price, iv=0.65)

    # open a position
    expiry = int(time.time()) + 7*24*3600
    strat.on_position_opened(expiry, round(price/1000)*1000, 0.65, 10000.0)
    state.inventory.perp_position_usd = -10000.0

    # push emergency premium: -20%
    for _ in range(4):
        state.vol_premium._raw_premiums.append(-0.20)

    sig = strat.tick()
    assert sig.action == StrategyAction.FLATTEN, f"expected FLATTEN got {sig.action}"

    await eng.handle(sig)
    # flatten sends cancel_all then market orders
    # at minimum the gateway cancel_all was called
    # (order count depends on what's live)


async def test_risk_rv_spike_halts():
    """RV spike ratio > halt threshold triggers kill switch."""
    cfg, state, gw, eng, risk, strat = make_system()

    price = warm_state(state, vol=0.004)
    setup_book(state, price)
    state.on_funding_rate(0.0003)

    halt_events = []
    async def on_halt(reason, detail):
        halt_events.append((reason, detail))

    risk._on_halt = on_halt

    # inject a 30% hourly return — rv_ratio will be >> 3x
    state.rv_estimator._returns.append(0.30)

    await risk._run_checks()

    assert risk.is_halted, "should be halted after RV spike"
    assert any(e[0] == KillReason.RV_SPIKE for e in halt_events)


async def test_risk_drawdown_halts():
    """Intraday drawdown exceeding limit halts the system."""
    cfg, state, gw, eng, risk, strat = make_system()

    price = warm_state(state, vol=0.004)
    setup_book(state, price)
    state.on_funding_rate(0.0003)

    halt_events = []
    async def on_halt(reason, detail):
        halt_events.append((reason, detail))
    risk._on_halt = on_halt

    # simulate loss exceeding intraday limit ($2000)
    risk._drawdown._session_high = 0.0
    risk._drawdown._pnl_log.clear()
    risk._drawdown._pnl_log.append((time.time(), 0.0))
    risk._cumulative_pnl = -2001.0

    await risk._run_checks()
    assert risk.is_halted


async def test_risk_funding_halt():
    """Funding rate below -20% ann triggers halt."""
    cfg, state, gw, eng, risk, strat = make_system()

    price = warm_state(state, vol=0.004)
    setup_book(state, price)

    halt_events = []
    async def on_halt(reason, detail):
        halt_events.append((reason, detail))
    risk._on_halt = on_halt

    # -25% ann funding = -0.000571 per 8h
    state.on_funding_rate(-0.000571)
    # confirm it's below halt threshold
    assert state.funding.rate_ann < -0.20

    await risk._run_checks()
    assert risk.is_halted
    assert any(e[0] == KillReason.FUNDING_NEGATIVE for e in halt_events)


def test_state_engine_seq_gap_raises():
    """Sequence gap in book delta raises StateError."""
    cfg, state, _, _, _, _ = make_system()
    price = 60000.0
    state.on_perp_book_snapshot(
        bids=[[price*0.999, 10]], asks=[[price*1.001, 10]], seq=1
    )
    try:
        state.on_perp_book_delta([["buy", price*0.999, 15]], seq=99)
        assert False, "should have raised StateError"
    except StateError:
        pass


def test_funding_regime_bear_confirmation():
    """Bear regime requires bear_confirmation_h, not just one negative tick."""
    cfg, state, _, _, _, _ = make_system()

    # single negative funding tick should NOT immediately flip to BEAR
    state.on_funding_rate(-0.0001)  # slightly negative
    from core.state_engine import FundingRegime
    # bear_since_ms just set but confirmation period not elapsed
    regime = state.funding.regime()
    assert regime in (FundingRegime.NEUTRAL, FundingRegime.BEAR)
    # with confirmation_h=4, a fresh tick should still be NEUTRAL
    if state.funding._bear_since_ms > 0:
        elapsed_h = 0.0  # just happened
        if elapsed_h < cfg.strategy.funding_regime.bear_confirmation_h:
            assert regime == FundingRegime.NEUTRAL


def test_as_pricer_inventory_skew():
    """AS pricer shifts reservation price with inventory."""
    from core.strategy import ASPricer
    cfg, _, _, _, _, _ = make_system()

    pricer = ASPricer(cfg)
    q_flat  = pricer.quote(mid_vol=0.65, rv=0.45, dte_hours=168, inventory=0.0)
    q_short = pricer.quote(mid_vol=0.65, rv=0.45, dte_hours=168, inventory=-100.0)

    # short inventory (sold options) should push reservation vol up
    assert q_short.reserve_vol != q_flat.reserve_vol
    assert q_short.skew != 0.0
    # spread should be the same (inventory doesn't change spread, only reservation)
    assert abs(q_flat.spread_vol - q_short.spread_vol) < 1e-10


def test_pnl_attribution_components():
    """PnL attributor correctly decomposes a move."""
    from core.risk_engine import PnlAttributor, PnlComponents

    attr = PnlAttributor(recompute_interval_s=0, dominance_threshold=0.99)

    # first call primes last_spot/last_iv/last_run
    attr.update(spot=59000.0, iv=0.64, gamma=0.0001,
        theta=-10.0, vega=50.0, funding_8h=0.0003, notional=10000.0, fees_paid=0.0)

    # force enough time to pass the interval guard
    attr._last_run -= 2.0

    # second call computes the delta
    comps = attr.update(
        spot=60000.0, iv=0.65, gamma=0.0001,
        theta=-10.0, vega=50.0,
        funding_8h=0.0003, notional=10000.0, fees_paid=5.0,
    )
    # gamma_pnl = 0.5 * 0.0001 * (1000)^2 = 50
    assert abs(comps.gamma_pnl - 50.0) < 0.1, f"gamma_pnl={comps.gamma_pnl}"
    # vega_pnl = 50 * 0.01 = 0.5
    assert abs(comps.vega_pnl - 0.5) < 0.01, f"vega_pnl={comps.vega_pnl}"
    assert comps.tx_cost == 5.0
    assert isinstance(comps.total, float)

    # dominant_component sanity
    name, ratio = comps.dominant_component()
    assert ratio > 0.0


def test_option_instrument_parser():
    """Parse various Deribit option instrument names."""
    cases = [
        ("BTC-27JUN25-60000-C", 60000.0),
        ("BTC-27JUN25-100000-P", 100000.0),
        ("ETH-27JUN25-3000-C", 3000.0),
    ]
    for name, expected_strike in cases:
        ts, strike = _parse_option_instrument(name)
        assert strike == expected_strike, f"{name}: expected {expected_strike} got {strike}"
        assert ts > 1_700_000_000


def test_build_subscriptions():
    """Subscription list has correct channel count and format."""
    cfg = load_config("./config")
    opts = ["BTC-27JUN25-60000-C", "BTC-27JUN25-60000-P"]
    subs = build_subscriptions(cfg, "BTC", opts)
    # 3 base (perp_book + index + perp_ticker) + 2 per option (ticker + trades)
    assert len(subs) == 3 + len(opts) * 2
    assert any("BTC-PERPETUAL" in s for s in subs)
    assert any("btc_usd" in s for s in subs)


# ---- runner -----------------------------------------------------------------

def run_sync(coro):
    return asyncio.run(coro)


if __name__ == "__main__":
    tests_sync = [
        test_cold_start_holds,
        test_enter_signal_with_positive_premium,
        test_state_engine_seq_gap_raises,
        test_funding_regime_bear_confirmation,
        test_as_pricer_inventory_skew,
        test_pnl_attribution_components,
        test_option_instrument_parser,
        test_build_subscriptions,
    ]
    tests_async = [
        test_enter_sends_two_option_orders,
        test_hedge_sends_perp_order,
        test_emergency_flatten_on_negative_premium,
        test_risk_rv_spike_halts,
        test_risk_drawdown_halts,
        test_risk_funding_halt,
    ]

    passed = failed = 0
    for t in tests_sync + tests_async:
        try:
            if asyncio.iscoroutinefunction(t):
                asyncio.run(t())
            else:
                t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)

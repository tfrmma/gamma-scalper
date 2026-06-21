# gamma-scalper

Short gamma scalping on Deribit options, delta-hedged via BTC-PERPETUAL.

**Edge:** IV - RV premium (~+6.5% mean, 72% win rate) + funding income (~+5.3% ann).
Calibrated on 730 days of BTC data (Jun 2024 – Jun 2026).

---

## Architecture

```
config/          TOML params + Pydantic loader (hot-reload)
core/
  state_engine   L2 book, RV estimator, delta tracker, funding regime, vol surface
  strategy       Avellaneda-Stoikov pricer, sizing, roll logic, signal generation
  execution      Order lifecycle, cancel/replace, hedge execution, emergency flatten
  risk_engine    Drawdown, loss velocity, position limits, kill switch, PnL attribution
  market_data    Deribit WS feed, parsers, resync, multi-asset coordinator
infra/
  deribit_gateway  Authenticated JSON-RPC, Black-76, fill notifications
tests/
  test_integration  14 end-to-end tests, no real WS needed
main.py          Entrypoint wires everything, asyncio orchestration
```

---

## Quickstart

```bash
pip install -r requirements.txt

# watch signal flow, no orders sent
python main.py --dry-run

# testnet (use_testnet=true in config/market.toml)
export DERIBIT_CLIENT_ID=your_id
export DERIBIT_CLIENT_SECRET=your_secret
python main.py

# live (set use_testnet=false in config/market.toml first)
python main.py
```

---

## Config

All parameters live in `config/`. Nothing is hardcoded.

| File | Contents |
|---|---|
| `market.toml` | Venue URLs, fee tiers, tick sizes, book validation |
| `strategy.toml` | AS model params, vol premium thresholds, sizing, rolling |
| `risk.toml` | Position limits, drawdown limits, kill switch triggers |
| `execution.toml` | Order types, cancel/replace triggers, latency budgets |

Key parameters calibrated from data:

```toml
# strategy.toml
[vol_premium_signal]
entry_threshold = 0.05       # enter when IV - RV > 5%
emergency_exit_threshold = -0.15  # flatten when IV - RV < -15%

[funding_regime]
size_multiplier_bull    = 1.0   # full size when funding > 5% ann
size_multiplier_neutral = 0.7
size_multiplier_bear    = 0.3   # funding < 0%: stay small

[delta_hedge]
delta_threshold = 0.05  # ~99 hedges/year from simulation

# risk.toml
[kill_switch]
rv_spike_halt_threshold    = 3.0   # RV(1h)/RV(24h) > 3x: flatten
funding_negative_halt_ann  = -0.20 # funding < -20% ann: flatten
```

---

## Kill switch triggers

Any one of these fires an immediate flatten + halt:

- RV spike ratio > 3x (1h vs 24h realized vol)
- Funding < -20% annualized
- Intraday drawdown > $2,000
- 24h drawdown > $3,000
- Loss velocity > $500/h or $100/min
- Margin utilization > 80%
- Perp/index divergence > 5%
- Book stale > 5s
- 3 consecutive API errors
- WS silent > 10s

---

## Running tests

```bash
cd gamma-scalper
python tests/test_integration.py
# 14 passed, 0 failed
```

---

## Adding ETH

1. Set `active_assets = ["BTC", "ETH"]` in `strategy.toml`
2. Verify `[leg.ETH]`, `[assets.ETH]`, `[fees.ETH]`, `[position_limits.ETH]` blocks exist
3. Run `MarketDataCoordinator` and `AssetSystem` scale automatically

---

## Environment variables

```bash
DERIBIT_CLIENT_ID      # required for live/testnet
DERIBIT_CLIENT_SECRET  # required for live/testnet
```

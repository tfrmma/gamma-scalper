# gamma-scalper

Short gamma scalping on Deribit BTC/ETH options, delta-hedged via perpetuals.

**Edge:** IV - RV premium (~+6.5% mean, 72% win rate) + funding income (~+5.3% ann).
Calibrated on 730 days of BTC data (Jun 2024 - Jun 2026).

---

## Architecture

```
config/
  market.toml       venue constants, fees, tick sizes
  strategy.toml     AS model, vol premium signal, sizing, rolling, OFI
  risk.toml         position limits, drawdown, kill switch triggers
  execution.toml    order types, cancel/replace, latency budgets
  loader.py         Pydantic validation + hot-reload

core/
  state_engine      L2 book + OFI, Yang-Zhang RV estimator, delta tracker,
                    funding regime, SVI + SABR vol surface
  strategy          AS pricer (SABR vanna-adjusted), MLE k calibration,
                    OFI entry filter, multi-leg position registry,
                    straddle/strangle support, roll detector,
                    regime-conditional Sharpe filter, calendar spread hedge
  execution         order lifecycle, simultaneous roll, iceberg support,
                    smart partial fills, queue position estimation
  risk_engine       real fill PnL, per-leg attribution, live margin from
                    exchange, Telegram/Slack alerting, kill switch
  market_data       Deribit WS feed, SVI/SABR refit on greeks, resync

infra/
  deribit_gateway   JSON-RPC auth, push fill notifications (no polling),
                    private WS reconnect with re-auth, Black-76 pricer,
                    account summary for margin monitor
  logging_setup     structured JSON logs (Datadog/Loki/Grafana), StatsD
                    gauges, rotating file handler

tests/
  test_integration  14 end-to-end tests, no real WS needed
main.py             asyncio orchestration, dry-run mode, graceful shutdown
```

---

## Quickstart

```bash
pip install -r requirements.txt

# signal flow only, no orders
python main.py --dry-run

# testnet (use_testnet=true in config/market.toml)
export DERIBIT_CLIENT_ID=your_id
export DERIBIT_CLIENT_SECRET=your_secret
python main.py

# live
# set use_testnet=false in config/market.toml, then:
python main.py
```

---

## Config

Nothing hardcoded. Every threshold, model parameter, and fee lives in `config/`.

| File | Contents |
|---|---|
| `market.toml` | Venue URLs, fee tiers, tick sizes, book validation |
| `strategy.toml` | AS model, vol premium, OFI filter, sizing, rolling, hedge vehicle |
| `risk.toml` | Position limits, drawdown windows, kill switch triggers |
| `execution.toml` | Order types, cancel/replace triggers, iceberg, latency budgets |

Key parameters calibrated from 730-day BTC dataset:

```toml
# strategy.toml
[vol_premium_signal]
entry_threshold          = 0.05    # IV - RV > 5% to enter
emergency_exit_threshold = -0.15   # flatten below -15%

[funding_regime]
size_multiplier_bull    = 1.0      # >5% ann funding: full size
size_multiplier_neutral = 0.7
size_multiplier_bear    = 0.3      # negative funding: stay small

[ofi]
entry_threshold = 0.60             # skip entry if |OFI| > 0.6

[delta_hedge]
delta_threshold = 0.05             # hedge when accumulated delta exceeds this

[realized_vol]
estimator = "yang_zhang"           # YZ: ~5-8x more efficient than C2C

# risk.toml
[kill_switch]
rv_spike_halt_threshold   = 3.0    # RV(1h)/RV(24h) > 3x: flatten
funding_negative_halt_ann = -0.20  # funding < -20% ann: flatten
```

Config hot-reloads every 5 minutes without restart.

---

## Strategy

**Entry conditions (all must hold):**
- IV - RV > `entry_threshold` (default 5%)
- `|OFI|` < 0.60 (no strong directional flow on the perp)
- Rolling 30-day Sharpe of vol premium > `sharpe_filter_threshold`
- Funding regime multiplier > 0 (not in confirmed bear regime)

**Pricing:**
Avellaneda-Stoikov in vol space. Reservation price adjusted for inventory skew
and SABR vanna (how much IV moves with spot). AS arrival rate `k` calibrated
via Poisson MLE from live fill data.

**Structures:**
- `straddle`: sell ATM call + put, same expiry
- `strangle`: sell OTM call + put at configurable delta targets (e.g. 25-delta)

**Hedge vehicle:**
- Default: `BTC-PERPETUAL` (funding income when positive)
- Fallback: quarterly futures when funding is persistently negative (calendar spread)

**Rolling:**
Three independent triggers: DTE < threshold, moneyness drift > 5%, vol surface
shift > 10%. Simultaneous roll — close and open fire concurrently via
`asyncio.gather`, no gap between legs.

**Multi-asset:**
ETH config is included. Add `"ETH"` to `active_assets` in `strategy.toml`
and the coordinator spins up a second feed + strategy loop automatically.

---

## Kill switch triggers

Any one fires an immediate flatten + halt + alert:

- RV spike > 3x (1h vs 24h)
- Funding < -20% annualized
- Intraday drawdown > $2,000
- 24h drawdown > $3,000
- Loss velocity > $500/h or $100/min
- Margin utilization > 80%
- Perp/index divergence > 5%
- Book stale > 5s
- 3 consecutive API errors
- Private WS silent > 10s

---

## Observability

**Logs:** structured JSON to stdout + rotating file. Compatible with Datadog,
Grafana Loki, and any collector that reads JSON lines.

**Metrics:** StatsD gauges emitted from every risk snapshot.
Set `STATSD_HOST` to enable.

**Alerts:** Telegram and/or Slack on halt.
Set `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` and/or `SLACK_WEBHOOK_URL`.

---

## Running tests

```bash
python tests/test_integration.py
# 14 passed, 0 failed
```

---

## Environment variables

```bash
# required
DERIBIT_CLIENT_ID
DERIBIT_CLIENT_SECRET

# optional - alerting
TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID
SLACK_WEBHOOK_URL

# optional - metrics
STATSD_HOST          # e.g. localhost
STATSD_PORT          # default 8125
```

from __future__ import annotations

import tomllib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

log = logging.getLogger("config")


# market.toml

class VenueRateLimits(BaseModel):
    max_requests_per_second:     int
    max_orders_per_second:       int
    websocket_reconnect_delay_s: float
    heartbeat_interval_s:        float


class VenueConfig(BaseModel):
    name:             str
    options_ws_url:   str
    rest_url:         str
    testnet_ws_url:   str
    testnet_rest_url: str
    use_testnet:      bool
    rate_limits:      VenueRateLimits

    @property
    def ws_url(self) -> str:
        return self.testnet_ws_url if self.use_testnet else self.options_ws_url

    @property
    def api_url(self) -> str:
        return self.testnet_rest_url if self.use_testnet else self.rest_url


class AssetConfig(BaseModel):
    currency:              str
    option_kind:           Literal["european", "american"]
    perp_instrument:       str
    index_instrument:      str
    tick_size_perp:        float
    min_trade_amount_perp: float
    contract_size:         float
    funding_interval_h:    int
    dvol_index:            str


class FeeConfig(BaseModel):
    option_maker_rebate: float
    option_taker_fee:    float
    option_delivery_fee: float
    perp_maker_fee:      float
    perp_taker_fee:      float
    settlement_fee:      float


class OrderbookConfig(BaseModel):
    max_spread_pct:          float
    min_bid_levels:          int
    min_ask_levels:          int
    stale_book_threshold_ms: int
    sequence_gap_action:     Literal["resync", "halt"]


class MarketConfig(BaseModel):
    venue:     VenueConfig
    assets:    dict[str, AssetConfig]
    fees:      dict[str, FeeConfig]
    orderbook: OrderbookConfig


# strategy.toml

class _StrategyMeta(BaseModel):
    active_assets: list[str]
    mode:          Literal["short_gamma", "long_gamma"]
    hedge_vehicle: Literal["perpetual"]


class LegConfig(BaseModel):
    structure:          Literal["straddle", "strangle", "call", "put"]
    target_dte_days:    int
    max_dte_days:       int
    min_dte_days:       int
    roll_dte_threshold: int
    moneyness_mode:     Literal["atm", "delta_target"]
    delta_target:       float
    max_moneyness_pct:  float
    base_notional_usd:  float
    max_notional_usd:   float
    min_notional_usd:   float
    strangle_call_delta: float = 0.25
    strangle_put_delta:  float = 0.25

    @model_validator(mode="after")
    def check_notional_bounds(self) -> "LegConfig":
        if not (self.min_notional_usd <= self.base_notional_usd <= self.max_notional_usd):
            raise ValueError(
                f"notional bounds broken: min={self.min_notional_usd} "
                f"base={self.base_notional_usd} max={self.max_notional_usd}"
            )
        return self


class AvellanedaStoikovConfig(BaseModel):
    gamma:                  float = Field(gt=0, lt=1)
    k:                      float = Field(gt=0)
    T_hours_default:        float = Field(gt=0)
    spread_min_vol_pts:     float = Field(gt=0)
    spread_max_vol_pts:     float
    skew_cap_vol_pts:       float = Field(gt=0)
    calibration_window_h:   int
    calibration_min_trades: int

    @model_validator(mode="after")
    def check_spread_bounds(self) -> "AvellanedaStoikovConfig":
        if self.spread_min_vol_pts >= self.spread_max_vol_pts:
            raise ValueError("spread_min_vol_pts must be < spread_max_vol_pts")
        return self


class RealizedVolConfig(BaseModel):
    primary_window_h:     int
    secondary_window_h:   int
    estimator:            Literal["close_to_close", "parkinson", "yang_zhang"]
    annualization_factor: int
    min_observations:     int


class VolPremiumSignalConfig(BaseModel):
    entry_threshold:          float
    exit_threshold:           float
    emergency_exit_threshold: float
    signal_smoothing_h:       int

    @model_validator(mode="after")
    def check_threshold_order(self) -> "VolPremiumSignalConfig":
        if not (self.emergency_exit_threshold < self.exit_threshold <= self.entry_threshold):
            raise ValueError(
                f"threshold order broken: emergency={self.emergency_exit_threshold} "
                f"exit={self.exit_threshold} entry={self.entry_threshold}"
            )
        return self


class FundingRegimeConfig(BaseModel):
    bull_threshold:          float
    bear_threshold:          float
    size_multiplier_bull:    float = Field(ge=0, le=1)
    size_multiplier_neutral: float = Field(ge=0, le=1)
    size_multiplier_bear:    float = Field(ge=0, le=1)
    bear_confirmation_h:     int

    @model_validator(mode="after")
    def check_multiplier_order(self) -> "FundingRegimeConfig":
        if not (self.size_multiplier_bear <= self.size_multiplier_neutral <= self.size_multiplier_bull):
            raise ValueError("multipliers must satisfy: bear <= neutral <= bull")
        return self


class DeltaHedgeConfig(BaseModel):
    delta_threshold:             float = Field(gt=0, lt=1)
    option_gamma_proxy:          float = Field(gt=0)
    max_hedge_interval_h:        float
    execution_style:             Literal["market", "aggressive_limit"]
    aggressive_limit_timeout_ms: int


class RollingConfig(BaseModel):
    roll_on_dte_below:             int
    roll_on_moneyness_pct:         float
    roll_on_vol_surface_shift_pct: float
    roll_style:                    Literal["simultaneous", "leg_by_leg"]
    roll_max_slippage_pct:         float


class OfiConfig(BaseModel):
    entry_threshold: float = Field(ge=0.0, le=1.0)


class SharpeFilterConfig(BaseModel):
    enabled:          bool
    window_h:         int
    min_sharpe:       float
    min_observations: int


class CalendarHedgeConfig(BaseModel):
    enabled:                   bool
    switch_funding_threshold:  float
    switch_confirmation_h:     int
    target_expiry_days:        int
    max_basis_pct:             float


class StrategyConfig(BaseModel):
    strategy:           _StrategyMeta
    leg:                dict[str, LegConfig]
    avellaneda_stoikov: AvellanedaStoikovConfig
    realized_vol:       RealizedVolConfig
    vol_premium_signal: VolPremiumSignalConfig
    sharpe_filter:      SharpeFilterConfig
    ofi:                OfiConfig
    funding_regime:     FundingRegimeConfig
    calendar_hedge:     CalendarHedgeConfig
    delta_hedge:        DeltaHedgeConfig
    rolling:            RollingConfig

    @model_validator(mode="after")
    def check_legs_defined(self) -> "StrategyConfig":
        for asset in self.strategy.active_assets:
            if asset not in self.leg:
                raise ValueError(f"active asset '{asset}' has no [leg.{asset}] block")
        return self


# risk.toml

class PositionLimitsAsset(BaseModel):
    max_vega_usd:           float
    max_gamma_usd:          float
    max_theta_usd_per_day:  float
    max_delta_notional_usd: float
    max_open_notional_usd:  float
    max_perp_notional_usd:  float
    max_delta_btc:          float | None = None
    max_delta_eth:          float | None = None


class PositionLimitsAggregate(BaseModel):
    max_total_notional_usd: float
    max_total_vega_usd:     float
    max_margin_utilization: float = Field(gt=0, lt=1)


class DrawdownConfig(BaseModel):
    max_intraday_drawdown_usd: float
    max_intraday_drawdown_pct: float = Field(gt=0, lt=1)
    max_drawdown_24h_usd:      float
    max_drawdown_7d_usd:       float
    max_drawdown_30d_usd:      float
    recovery_threshold_usd:    float
    recovery_wait_h:           int


class KillSwitchConfig(BaseModel):
    api_error_consecutive:          int
    api_latency_threshold_ms:       int
    api_latency_consecutive:        int
    websocket_disconnect_max_s:     int
    orderbook_stale_max_s:          int
    index_divergence_pct:           float
    sequence_gap_max:               int
    rv_spike_threshold:             float
    rv_spike_halt_threshold:        float
    funding_negative_halt_ann:      float
    premium_negative_consecutive_h: int
    premium_emergency_h:            int
    max_loss_per_hour_usd:          float
    max_loss_per_minute_usd:        float


class MarginConfig(BaseModel):
    min_margin_buffer_pct:       float = Field(gt=0, lt=1)
    margin_warning_pct:          float
    margin_reduce_pct:           float
    margin_halt_pct:             float
    emergency_reduce_step_pct:   float
    emergency_reduce_interval_s: int


class PnlAttributionConfig(BaseModel):
    min_confidence_pct:            float
    recompute_interval_s:          int
    component_dominance_threshold: float


class RiskConfig(BaseModel):
    position_limits: dict[str, Any]
    drawdown:        DrawdownConfig
    kill_switch:     KillSwitchConfig
    margin:          MarginConfig
    pnl_attribution: PnlAttributionConfig

    def asset_limits(self, asset: str) -> PositionLimitsAsset:
        raw = self.position_limits.get(asset)
        if raw is None:
            raise KeyError(f"no position limits for '{asset}' in risk.toml")
        return PositionLimitsAsset(**raw)

    @property
    def aggregate_limits(self) -> PositionLimitsAggregate:
        return PositionLimitsAggregate(**self.position_limits["aggregate"])


# execution.toml

class OptionsExecutionConfig(BaseModel):
    order_type:                 Literal["limit"]
    post_only:                  bool
    price_improvement_vol_pts:  float
    cancel_on_iv_drift_vol_pts: float
    cancel_on_index_move_pct:   float
    cancel_on_time_ms:          int
    min_fill_size_pct:          float
    self_trade_prevention:      Literal["cancel_newest", "cancel_oldest", "reject"]
    max_order_retries:          int
    retry_delay_ms:             int

    @field_validator("post_only")
    @classmethod
    def must_be_post_only(cls, v: bool) -> bool:
        if not v:
            raise ValueError("post_only=false? no. taker fees eat the entire edge.")
        return v


class PerpExecutionConfig(BaseModel):
    order_type:             Literal["aggressive_limit", "market"]
    limit_offset_ticks:     int
    convert_to_market_ms:   int
    max_slippage_ticks:     int
    max_slippage_pct:       float
    round_to_contract_size: bool
    emergency_order_type:   Literal["market"]
    emergency_max_retries:  int


class LatencyConfig(BaseModel):
    target_market_data_lag_ms:  int
    target_signal_compute_ms:   int
    target_order_send_ms:       int
    target_hedge_roundtrip_ms:  int
    hard_limit_order_send_ms:   int
    hard_limit_hedge_ms:        int
    measurement_window_s:       int
    alert_percentile:           int


class FootprintConfig(BaseModel):
    spread_across_levels: bool
    max_visible_size_pct: float
    timing_jitter_ms:     int
    size_jitter_pct:      float
    use_iceberg:          bool


class ExecutionConfig(BaseModel):
    options_execution: OptionsExecutionConfig
    perp_execution:    PerpExecutionConfig
    latency:           LatencyConfig
    footprint:         FootprintConfig


# top-level container

@dataclass
class Config:
    market:      MarketConfig
    strategy:    StrategyConfig
    risk:        RiskConfig
    execution:   ExecutionConfig
    _config_dir: Path = field(repr=False)

    def active_assets(self) -> list[str]:
        return self.strategy.strategy.active_assets

    def reload(self) -> None:
        """Re-read all files from disk. Useful for param tweaks without restart."""
        log.info("reloading config...")
        fresh = _load_from_dir(self._config_dir)
        self.market    = fresh.market
        self.strategy  = fresh.strategy
        self.risk      = fresh.risk
        self.execution = fresh.execution
        log.info("config reloaded ok")


def _read_toml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"missing config: {path}")
    with open(path, "rb") as f:
        return tomllib.load(f)


def _validate_cross_file(strategy: StrategyConfig, market: MarketConfig) -> None:
    # every active asset needs [assets.X] and [fees.X] in market.toml
    for asset in strategy.strategy.active_assets:
        if asset not in market.assets:
            raise ValueError(f"'{asset}' in active_assets but missing [assets.{asset}] in market.toml")
        if asset not in market.fees:
            raise ValueError(f"'{asset}' in active_assets but missing [fees.{asset}] in market.toml")


def _load_from_dir(config_dir: Path) -> Config:
    market    = MarketConfig(**_read_toml(config_dir / "market.toml"))
    strategy  = StrategyConfig(**_read_toml(config_dir / "strategy.toml"))
    risk      = RiskConfig(**_read_toml(config_dir / "risk.toml"))
    execution = ExecutionConfig(**_read_toml(config_dir / "execution.toml"))

    _validate_cross_file(strategy, market)

    return Config(
        market=market,
        strategy=strategy,
        risk=risk,
        execution=execution,
        _config_dir=config_dir,
    )


def load_config(config_dir: str | Path = "./config") -> Config:
    config_dir = Path(config_dir)
    log.info(f"loading config from {config_dir.resolve()}")
    cfg = _load_from_dir(config_dir)
    log.info(
        f"config ok | mode={cfg.strategy.strategy.mode} "
        f"assets={cfg.active_assets()} testnet={cfg.market.venue.use_testnet}"
    )
    return cfg

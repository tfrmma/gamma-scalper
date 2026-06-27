from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import time
from pathlib import Path

from config.loader import load_config, Config
from core.state_engine import StateEngine
from core.strategy import GammaScalpStrategy, StrategyAction
from core.execution import ExecutionEngine, ExchangeGateway
from core.risk_engine import RiskEngine, KillReason
from core.market_data import MarketDataCoordinator
from infra.deribit_gateway import DeribitGateway
from infra.logging_setup import setup_logging


log = logging.getLogger("main")


# ---- stub gateway (dry-run only) --------------------------------------------

class _StubGateway(ExchangeGateway):
    """
    Logs everything, sends nothing.
    Used when --dry-run is passed - lets you watch signal flow without touching the exchange.
    """
    def __init__(self) -> None:
        self._log = logging.getLogger("stub_gateway")

    async def send_order(self, order) -> dict:
        self._log.info(f"STUB send | {order.instrument} {order.side} {order.size} @ {order.price}")
        return {"id": f"STUB_{int(time.monotonic()*1000)}"}

    async def cancel_order(self, exchange_id: str) -> dict:
        self._log.info(f"STUB cancel | {exchange_id}")
        return {"result": "ok"}

    async def cancel_all(self, instrument: str | None = None) -> dict:
        self._log.info(f"STUB cancel_all | instrument={instrument}")
        return {"result": "ok"}

    async def get_order(self, exchange_id: str) -> dict:
        return {"order_state": "filled", "filled_amount": 1.0, "average_price": 0.0}

    async def get_positions(self) -> dict:
        return {}


# ---- per-asset system -------------------------------------------------------

class AssetSystem:
    def __init__(self, cfg: Config, asset: str, gateway: ExchangeGateway) -> None:
        self.asset = asset
        self.cfg   = cfg

        self.state     = StateEngine(cfg, asset)
        self.execution = ExecutionEngine(cfg, self.state, gateway, asset,
                                         on_fill=self._on_fill)
        self.risk      = RiskEngine(cfg, self.state, self.execution, asset,
                                    on_halt=self._on_halt)
        self.strategy  = GammaScalpStrategy(cfg, self.state, asset)

    async def _on_fill(self, exchange_id: str, size: float, price: float, greek: float) -> None:
        log.info(f"fill | {self.asset} | eid={exchange_id} size={size} px={price:.2f}")
        self.strategy.on_fill()
        self.risk.record_pnl_tick(0.0)   # TODO: compute real PnL from fill + greeks

    async def _on_halt(self, reason: KillReason, detail: str) -> None:
        log.error(f"HALT | {self.asset} | {reason.name} | {detail}")
        # TODO: wire up alerting here (telegram, pagerduty, etc.)


# ---- loops ------------------------------------------------------------------

async def strategy_loop(system: AssetSystem, tick_interval_s: float = 0.5) -> None:
    log.info(f"strategy loop started | asset={system.asset}")
    while True:
        try:
            if system.risk.is_halted:
                await asyncio.sleep(tick_interval_s)
                continue

            sig = system.strategy.tick()

            if sig.action != StrategyAction.HOLD:
                log.info(f"signal | {system.asset} | {sig.action.name} | {sig.reason}")

            # keep gateway spot price current for vol->price conversion
            if isinstance(system.execution.gateway, DeribitGateway):
                try:
                    system.execution.gateway.update_spot(system.state.spot())
                except Exception:
                    pass

            await system.execution.handle(sig)

        except asyncio.CancelledError:
            log.info(f"strategy loop cancelled | asset={system.asset}")
            break
        except Exception as e:
            log.error(f"strategy loop error: {e}", exc_info=True)
            system.risk.record_api_error()

        await asyncio.sleep(tick_interval_s)


async def snapshot_loop(
    systems:    dict[str, AssetSystem],
    feed:       MarketDataCoordinator,
    interval_s: float = 60.0,
) -> None:
    while True:
        try:
            await asyncio.sleep(interval_s)
            for asset, sys_ in systems.items():
                log.info(f"state    | {sys_.state.snapshot()}")
                log.info(f"strategy | {sys_.strategy.status()}")
                log.info(f"risk     | {sys_.risk.snapshot()}")
                log.info(f"exec     | {sys_.execution.snapshot()}")
            log.info(f"feed     | {feed.snapshot()}")
        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error(f"snapshot error: {e}")


async def config_reload_loop(cfg: Config, interval_s: float = 300.0) -> None:
    while True:
        try:
            await asyncio.sleep(interval_s)
            cfg.reload()
        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error(f"config reload failed: {e}")


# ---- shutdown ---------------------------------------------------------------

class ShutdownHandler:
    def __init__(self) -> None:
        self._event = asyncio.Event()

    def install(self, loop: asyncio.AbstractEventLoop) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._trigger)

    def _trigger(self) -> None:
        log.warning("shutdown signal received")
        self._event.set()

    async def wait(self) -> None:
        await self._event.wait()


# ---- gateway factory --------------------------------------------------------

async def _build_gateway(
    cfg:      Config,
    asset:    str,
    dry_run:  bool,
    on_fill:  object,
) -> ExchangeGateway:
    """
    Build and connect the right gateway.
    dry_run=True: stub, no network.
    dry_run=False: real Deribit gateway, authenticated, subscribed to fills.
    """
    if dry_run:
        log.warning("dry-run mode - stub gateway, no orders sent")
        return _StubGateway()

    gw = DeribitGateway.from_env(cfg, asset=asset)
    await gw.connect()
    await gw.authenticate()
    gw.set_fill_callback(on_fill)
    await gw.subscribe_private()
    log.info(f"gateway connected and authenticated | asset={asset}")
    return gw


# ---- main -------------------------------------------------------------------

async def run(cfg: Config, assets: list[str], dry_run: bool) -> None:
    shutdown = ShutdownHandler()
    shutdown.install(asyncio.get_running_loop())

    # build systems first (we need on_fill callback before building gateway)
    # use stub temporarily, swap to real gateway after auth
    systems: dict[str, AssetSystem] = {}
    for asset in assets:
        log.info(f"initializing | {asset}")
        stub = _StubGateway()
        systems[asset] = AssetSystem(cfg, asset, stub)

    # now build real gateways and rewire execution engines
    gateways: dict[str, ExchangeGateway] = {}
    for asset, sys_ in systems.items():
        gw = await _build_gateway(cfg, asset, dry_run, sys_._on_fill)
        gateways[asset] = gw
        # rewire execution engine to use the real gateway
        sys_.execution.gateway = gw

    # give gateway a handle to spot price (needed for vol->price on option sends)
    def _spot_updater(asset: str, gw: ExchangeGateway):
        async def _update():
            while True:
                try:
                    if isinstance(gw, DeribitGateway):
                        gw.update_spot(systems[asset].state.spot())
                except Exception:
                    pass
                await asyncio.sleep(1.0)
        return _update

    feed = MarketDataCoordinator(
        cfg    = cfg,
        states = {a: s.state for a, s in systems.items()},
        risks  = {a: s.risk  for a, s in systems.items()},
    )

    tasks: list[asyncio.Task] = []
    tasks.append(asyncio.create_task(feed.start(), name="feed"))

    for asset, sys_ in systems.items():
        tasks.append(asyncio.create_task(sys_.risk.run(),        name=f"risk_{asset}"))
        tasks.append(asyncio.create_task(strategy_loop(sys_),    name=f"strategy_{asset}"))
        if isinstance(gateways[asset], DeribitGateway):
            tasks.append(asyncio.create_task(
                _spot_updater(asset, gateways[asset])(),
                name=f"spot_updater_{asset}",
            ))

    tasks.append(asyncio.create_task(snapshot_loop(systems, feed), name="snapshot"))
    tasks.append(asyncio.create_task(config_reload_loop(cfg),      name="config_reload"))

    log.info(f"system up | assets={assets} testnet={cfg.market.venue.use_testnet} dry_run={dry_run}")
    log.info(f"mode={cfg.strategy.strategy.mode} | {len(tasks)} tasks")

    shutdown_task = asyncio.create_task(shutdown.wait(), name="shutdown")
    done, pending = await asyncio.wait(
        [shutdown_task, *tasks],
        return_when=asyncio.FIRST_COMPLETED,
    )

    for t in done:
        if t is not shutdown_task:
            exc = t.exception()
            if exc:
                log.error(f"task {t.get_name()} crashed: {exc}", exc_info=exc)

    log.warning("shutting down - flattening all positions")
    for asset, sys_ in systems.items():
        if not sys_.risk.is_halted:
            try:
                await sys_.execution._emergency_flatten(reason="shutdown")
            except Exception as e:
                log.error(f"flatten failed for {asset}: {e}")

    # disconnect real gateways cleanly
    for gw in gateways.values():
        if isinstance(gw, DeribitGateway):
            try:
                await gw.disconnect()
            except Exception:
                pass

    for t in pending:
        t.cancel()
    await asyncio.gather(*pending, return_exceptions=True)

    await feed.stop()
    log.info("shutdown complete")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gamma scalping system - Pink Panthers")
    parser.add_argument("--config",    default="./config", help="Config directory")
    parser.add_argument("--asset",     default=None,       help="Override active assets (e.g. ETH)")
    parser.add_argument("--log-level", default="INFO",     help="Logging level")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Stub gateway - watch signal flow without touching the exchange")
    args = parser.parse_args()

    setup_logging(args.log_level)

    cfg    = load_config(args.config)
    assets = [args.asset] if args.asset else cfg.active_assets()
    if cfg.market.venue.use_testnet:
        log.warning(f"TESTNET | assets={assets} dry_run={args.dry_run}")
    else:
        log.warning(f"LIVE MODE | assets={assets} | orders will be sent to Deribit")
        if not args.dry_run:
            log.warning("you have 3 seconds to Ctrl-C")
            time.sleep(3)

    asyncio.run(run(cfg, assets, dry_run=args.dry_run))


if __name__ == "__main__":
    main()

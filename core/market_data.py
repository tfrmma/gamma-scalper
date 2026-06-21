from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

try:
    import websockets
    from websockets.exceptions import ConnectionClosed, WebSocketException
except ImportError:
    raise ImportError("pip install websockets")

from config.loader import Config
from core.state_engine import StateEngine, StateError
from core.risk_engine import RiskEngine

log = logging.getLogger("market_data")


# ---- connection state -------------------------------------------------------

class FeedStatus(Enum):
    DISCONNECTED = auto()
    CONNECTING   = auto()
    CONNECTED    = auto()
    SUBSCRIBING  = auto()
    LIVE         = auto()
    RESYNCING    = auto()


# ---- subscription builder ---------------------------------------------------

def _perp_book_channel(instrument: str) -> str:
    # group=1 = raw L2, depth=20 levels
    return f"book.{instrument}.none.20.100ms"

def _index_channel(index_name: str) -> str:
    return f"deribit_price_index.{index_name}"

def _funding_channel(instrument: str) -> str:
    return f"ticker.{instrument}.100ms"

def _option_channel(instrument: str) -> str:
    # ticker gives us IV + greeks on each option
    return f"ticker.{instrument}.100ms"

def _trades_channel(instrument: str) -> str:
    return f"trades.{instrument}.100ms"


def build_subscriptions(cfg: Config, asset: str, option_instruments: list[str]) -> list[str]:
    """
    Build the full list of channels to subscribe on connect/resync.
    option_instruments: list of Deribit option names we're currently watching,
                        e.g. ["BTC-27JUN25-60000-C", "BTC-27JUN25-60000-P"]
    """
    ac   = cfg.market.assets[asset]
    subs = [
        _perp_book_channel(ac.perp_instrument),
        _index_channel(ac.index_instrument),
        _funding_channel(ac.perp_instrument),
    ]
    for inst in option_instruments:
        subs.append(_option_channel(inst))
        subs.append(_trades_channel(inst))
    return subs


# ---- message parsers --------------------------------------------------------
# one function per channel type - keeps the dispatcher clean
# Deribit JSON shapes are documented at docs.deribit.com/v2

def _parse_book(data: dict, state: StateEngine) -> None:
    """
    book.* channel - L2 snapshot or delta.
    type: "snapshot" on first message, "change" on subsequent.
    """
    msg_type = data.get("type")
    seq      = data.get("change_id", 0)

    if msg_type == "snapshot":
        bids = data.get("bids", [])
        asks = data.get("asks", [])
        instrument = data.get("instrument_name", "")
        if "PERPETUAL" in instrument:
            state.on_perp_book_snapshot(bids, asks, seq)
        else:
            state.on_option_book_snapshot(bids, asks, seq)
    elif msg_type == "change":
        changes = []
        for side_key, side_label in [("bids", "buy"), ("asks", "sell")]:
            for entry in data.get(side_key, []):
                # Deribit delta format: [action, price, amount]
                # action: "new" | "change" | "delete"
                action, price, amount = entry
                size = 0.0 if action == "delete" else float(amount)
                changes.append([side_label, float(price), size])
        instrument = data.get("instrument_name", "")
        try:
            if "PERPETUAL" in instrument:
                state.on_perp_book_delta(changes, seq)
            else:
                state.on_option_book_delta(changes, seq)
        except StateError as e:
            # seq gap - caller handles resync
            raise


def _parse_index(data: dict, state: StateEngine) -> None:
    price = data.get("price")
    if price is not None:
        state.on_index_price(float(price))


def _parse_ticker(data: dict, state: StateEngine, asset: str) -> None:
    """
    ticker.* channel - used for both perp (funding) and options (greeks/IV).
    Deribit bundles everything in one message, so we pick what we need.
    """
    instrument = data.get("instrument_name", "")

    if "PERPETUAL" in instrument:
        funding = data.get("current_funding")
        if funding is not None:
            state.on_funding_rate(float(funding))
        return

    # option ticker - greeks are under "greeks" key
    greeks  = data.get("greeks", {})
    iv      = data.get("mark_iv")       # annualized IV as a fraction (e.g. 0.65)
    delta   = greeks.get("delta")
    gamma   = greeks.get("gamma")
    vega    = greeks.get("vega")
    theta   = greeks.get("theta")

    if None in (iv, delta, gamma, vega, theta):
        return  # incomplete greeks snapshot, skip

    # parse expiry and strike from instrument name: "BTC-27JUN25-60000-C"
    try:
        expiry_ts, strike = _parse_option_instrument(instrument)
    except ValueError:
        log.debug(f"could not parse option instrument: {instrument}")
        return

    state.on_greeks(
        expiry_ts = expiry_ts,
        strike    = float(strike),
        iv        = float(iv) / 100.0,   # Deribit sends IV as % (65.0), we want 0.65
        delta     = float(delta),
        gamma     = float(gamma),
        vega      = float(vega),
        theta     = float(theta),
    )


def _parse_trades(data: list, state: StateEngine, risk: RiskEngine) -> None:
    """
    trades.* channel - used to track fill arrival rate for AS k calibration.
    We don't process the fills here (those come via private WS), just the timestamps.
    """
    for trade in data:
        ts = trade.get("timestamp", 0) / 1000.0   # ms -> s
        # risk engine records api success on live trade data - proves feed is alive
        risk.record_api_success()


def _parse_option_instrument(name: str) -> tuple[int, float]:
    """
    Parse "BTC-27JUN25-60000-C" -> (expiry_unix_ts, 60000.0)
    Month map is annoying but there's no cleaner way without a datetime parse.
    """
    import datetime
    parts = name.split("-")
    if len(parts) != 4:
        raise ValueError(f"unexpected format: {name}")

    expiry_str = parts[1]   # "27JUN25"
    strike_str = parts[2]   # "60000"

    month_map = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
        "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
        "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }
    day   = int(expiry_str[:2])
    month = month_map[expiry_str[2:5]]
    year  = 2000 + int(expiry_str[5:])

    # Deribit options expire at 08:00 UTC
    expiry_dt = datetime.datetime(year, month, day, 8, 0, 0, tzinfo=datetime.timezone.utc)
    return int(expiry_dt.timestamp()), float(strike_str)


# ---- resync state machine ---------------------------------------------------

@dataclass
class ResyncState:
    """Tracks resync attempts per asset. Don't loop forever."""
    max_attempts:    int   = 5
    backoff_base_s:  float = 2.0
    attempts:        int   = 0
    last_attempt_ms: int   = 0

    def should_resync(self) -> bool:
        return self.attempts < self.max_attempts

    def record_attempt(self) -> float:
        self.attempts       += 1
        self.last_attempt_ms = _now_ms()
        delay = self.backoff_base_s * (2 ** (self.attempts - 1))
        return min(delay, 60.0)   # cap at 60s

    def reset(self) -> None:
        self.attempts        = 0
        self.last_attempt_ms = 0


# ---- main feed handler ------------------------------------------------------

class DeribitFeedHandler:
    """
    Manages the Deribit WebSocket connection for one asset.
    Handles connect, subscribe, dispatch, heartbeat, resync.

    One instance per asset. If you're running BTC + ETH, you have two of these.
    They share nothing except the event loop.

    Not doing any auth here - public channels only for market data.
    Private channels (fills, account) go through a separate authenticated connection
    in infra/deribit_gateway.py. Clean separation.
    """

    def __init__(
        self,
        cfg:                Config,
        state:              StateEngine,
        risk:               RiskEngine,
        asset:              str,
        option_instruments: list[str] | None = None,
    ) -> None:
        self.cfg   = cfg
        self.state = state
        self.risk  = risk
        self.asset = asset

        self._option_instruments: list[str] = option_instruments or []
        self._status  = FeedStatus.DISCONNECTED
        self._ws      = None
        self._resync  = ResyncState(
            max_attempts   = 5,
            backoff_base_s = cfg.market.venue.rate_limits.websocket_reconnect_delay_s,
        )
        self._running          = False
        self._last_heartbeat   = 0.0
        self._heartbeat_interval = cfg.market.venue.rate_limits.heartbeat_interval_s
        self._msg_count        = 0
        self._seq_errors       = 0

        self._ws_url = cfg.market.venue.ws_url
        log.info(f"feed handler ready | asset={asset} url={self._ws_url}")

    # ---- public -------------------------------------------------------------

    def update_option_instruments(self, instruments: list[str]) -> None:
        """Call when we roll to a new expiry - updates the subscription list."""
        self._option_instruments = instruments

    @property
    def status(self) -> FeedStatus:
        return self._status

    @property
    def is_live(self) -> bool:
        return self._status == FeedStatus.LIVE

    async def run(self) -> None:
        """Main loop. Run as an asyncio task. Reconnects on disconnect."""
        self._running = True
        while self._running:
            try:
                await self._connect_and_run()
                # clean exit - don't reconnect
                break
            except (ConnectionClosed, WebSocketException) as e:
                log.warning(f"WS disconnected: {e}")
                self._status = FeedStatus.DISCONNECTED
                self.risk.record_api_error()

                if not self._resync.should_resync():
                    log.error("max reconnect attempts reached - halting")
                    await self.risk.manual_halt("feed: max reconnect attempts")
                    break

                delay = self._resync.record_attempt()
                log.info(f"reconnecting in {delay:.1f}s (attempt {self._resync.attempts})")
                await asyncio.sleep(delay)

            except asyncio.CancelledError:
                log.info(f"feed handler cancelled | asset={self.asset}")
                break
            except Exception as e:
                log.error(f"unexpected feed error: {e}", exc_info=True)
                self.risk.record_api_error()
                await asyncio.sleep(5.0)

    async def stop(self) -> None:
        self._running = False
        if self._ws is not None:
            await self._ws.close()

    # ---- connection lifecycle -----------------------------------------------

    async def _connect_and_run(self) -> None:
        self._status = FeedStatus.CONNECTING
        log.info(f"connecting to {self._ws_url}")

        async with websockets.connect(
            self._ws_url,
            ping_interval = self._heartbeat_interval,
            ping_timeout  = self._heartbeat_interval * 2,
            max_size      = 2 ** 23,   # 8MB - options snapshots can be chunky
        ) as ws:
            self._ws     = ws
            self._status = FeedStatus.CONNECTED
            self._resync.reset()
            log.info(f"connected | asset={self.asset}")

            await self._subscribe()
            self._status = FeedStatus.LIVE
            self.risk.record_ws_heartbeat()

            # run message loop and heartbeat concurrently
            await asyncio.gather(
                self._message_loop(ws),
                self._heartbeat_loop(ws),
            )

    async def _subscribe(self) -> None:
        if self._ws is None:
            return

        self._status = FeedStatus.SUBSCRIBING
        channels     = build_subscriptions(self.cfg, self.asset, self._option_instruments)

        msg = {
            "jsonrpc": "2.0",
            "id":      1,
            "method":  "public/subscribe",
            "params":  {"channels": channels},
        }
        await self._ws.send(json.dumps(msg))
        log.info(f"subscribed to {len(channels)} channels | asset={self.asset}")

        # don't wait for ack here - messages start flowing immediately
        # the first book message will be a snapshot, which resets state correctly

    async def _resubscribe(self) -> None:
        """Called after a seq gap - re-subscribe to get fresh snapshots."""
        self._status = FeedStatus.RESYNCING
        log.info(f"resyncing | asset={self.asset}")
        self._seq_errors += 1

        if self._seq_errors > self.cfg.market.orderbook.sequence_gap_max:
            await self.risk.manual_halt(f"feed: {self._seq_errors} seq gaps on {self.asset}")
            return

        await self._subscribe()
        self._status = FeedStatus.LIVE

    # ---- message loop -------------------------------------------------------

    async def _message_loop(self, ws) -> None:
        async for raw in ws:
            try:
                await self._dispatch(json.loads(raw))
                self._msg_count += 1
                # every 100 messages, tell risk engine the feed is alive
                if self._msg_count % 100 == 0:
                    self.risk.record_ws_heartbeat()
                    self.risk.record_api_success()
            except StateError as e:
                # seq gap - need resync
                log.warning(f"state error in dispatch: {e}")
                await self._resubscribe()
            except json.JSONDecodeError as e:
                log.error(f"bad JSON from WS: {e}")
                self.risk.record_api_error()
            except Exception as e:
                log.error(f"dispatch error: {e}", exc_info=True)
                # don't kill the loop on a single bad message

    async def _dispatch(self, msg: dict) -> None:
        """
        Route a parsed WS message to the right parser.
        Deribit wraps everything in {"method": "subscription", "params": {"channel": ..., "data": ...}}
        """
        method = msg.get("method")

        # subscription data
        if method == "subscription":
            params  = msg.get("params", {})
            channel = params.get("channel", "")
            data    = params.get("data", {})
            await self._route_channel(channel, data)
            return

        # heartbeat from server
        if method == "heartbeat":
            await self._ws.send(json.dumps({
                "jsonrpc": "2.0",
                "id":      9999,
                "method":  "public/test",
                "params":  {},
            }))
            self.risk.record_ws_heartbeat()
            return

        # subscription ack or other RPC response - ignore
        if "result" in msg or "error" in msg:
            if "error" in msg:
                log.warning(f"WS error response: {msg['error']}")
                self.risk.record_api_error()
            return

    async def _route_channel(self, channel: str, data: Any) -> None:
        """
        Map channel name prefix to parser.
        Order matters: check more-specific prefixes first.
        """
        if channel.startswith("book."):
            _parse_book(data, self.state)

        elif channel.startswith("deribit_price_index."):
            _parse_index(data, self.state)

        elif channel.startswith("ticker."):
            _parse_ticker(data, self.state, self.asset)

        elif channel.startswith("trades."):
            _parse_trades(data if isinstance(data, list) else [data], self.state, self.risk)

        else:
            log.debug(f"unhandled channel: {channel}")

    # ---- heartbeat loop -----------------------------------------------------

    async def _heartbeat_loop(self, ws) -> None:
        """
        Send a test request every heartbeat_interval seconds.
        Deribit will close the connection if it doesn't hear from us.
        We also use this to measure round-trip latency on the WS.
        """
        while True:
            await asyncio.sleep(self._heartbeat_interval)
            t0 = _now_ms()
            try:
                await ws.send(json.dumps({
                    "jsonrpc": "2.0",
                    "id":      8888,
                    "method":  "public/test",
                    "params":  {},
                }))
                self._last_heartbeat = time.monotonic()
                self.risk.record_ws_heartbeat()
            except Exception as e:
                log.warning(f"heartbeat failed: {e}")
                self.risk.record_api_error()
                break   # let the outer loop handle reconnect

    def snapshot(self) -> dict:
        return {
            "asset":      self.asset,
            "status":     self._status.name,
            "msg_count":  self._msg_count,
            "seq_errors": self._seq_errors,
            "resync_attempts": self._resync.attempts,
            "last_heartbeat_s": (time.monotonic() - self._last_heartbeat)
                                  if self._last_heartbeat > 0 else -1,
        }


# ---- multi-asset coordinator ------------------------------------------------

class MarketDataCoordinator:
    """
    One feed handler per active asset. Starts/stops them together.
    Also owns the option instrument list - updated when strategy rolls.
    """

    def __init__(
        self,
        cfg:    Config,
        states: dict[str, StateEngine],
        risks:  dict[str, RiskEngine],
    ) -> None:
        self.cfg     = cfg
        self._handlers: dict[str, DeribitFeedHandler] = {}
        self._tasks:    dict[str, asyncio.Task]        = {}

        for asset in cfg.active_assets():
            if asset not in states or asset not in risks:
                raise ValueError(f"no state/risk engine for asset {asset}")
            self._handlers[asset] = DeribitFeedHandler(
                cfg   = cfg,
                state = states[asset],
                risk  = risks[asset],
                asset = asset,
            )

    def update_options(self, asset: str, instruments: list[str]) -> None:
        """Call after a roll to update which option channels we subscribe to."""
        if asset in self._handlers:
            self._handlers[asset].update_option_instruments(instruments)

    async def start(self) -> None:
        for asset, handler in self._handlers.items():
            task = asyncio.create_task(handler.run(), name=f"feed_{asset}")
            self._tasks[asset] = task
            log.info(f"started feed task | asset={asset}")

    async def stop(self) -> None:
        for handler in self._handlers.values():
            await handler.stop()
        for task in self._tasks.values():
            task.cancel()
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        log.info("all feed handlers stopped")

    def is_live(self) -> bool:
        return all(h.is_live for h in self._handlers.values())

    def snapshot(self) -> dict:
        return {asset: h.snapshot() for asset, h in self._handlers.items()}


# ---- util -------------------------------------------------------------------

def _now_ms() -> int:
    return int(time.monotonic() * 1000)

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Awaitable

try:
    import websockets
    from websockets.exceptions import ConnectionClosed, WebSocketException
except ImportError:
    raise ImportError("pip install websockets")

from config.loader import Config
from core.execution import ExchangeGateway, Order

log = logging.getLogger("gateway")


# ---- Black-76 + CDF ---------------------------------------------------------

def _norm_cdf(x: float) -> float:
    if x < 0:
        return 1.0 - _norm_cdf(-x)
    t = 1.0 / (1.0 + 0.2316419 * x)
    poly = t * (0.319381530
              + t * (-0.356563782
              + t * (1.781477937
              + t * (-1.821255978
              + t * 1.330274429))))
    return 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x) * poly


def _black76_price(F: float, K: float, T: float, sigma: float, r: float, is_call: bool) -> float:
    if T <= 0 or sigma <= 0:
        intrinsic = max(0.0, F - K) if is_call else max(0.0, K - F)
        return intrinsic / F
    sqrt_T = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    disc = math.exp(-r * T)
    if is_call:
        return disc * (F * _norm_cdf(d1) - K * _norm_cdf(d2)) / F
    return disc * (K * _norm_cdf(-d2) - F * _norm_cdf(-d1)) / F


def vol_to_price(vol: float, spot: float, strike: float, expiry_ts: int, is_call: bool) -> float:
    T = max(0.0, (expiry_ts - time.time()) / (365 * 24 * 3600))
    return max(_black76_price(spot, strike, T, vol, 0.0, is_call), 0.0001)


# ---- order serialization ----------------------------------------------------

def _parse_expiry(s: str) -> int:
    import datetime
    month_map = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
        "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
        "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }
    day, month, year = int(s[:2]), month_map[s[2:5]], 2000 + int(s[5:])
    dt = datetime.datetime(year, month, day, 8, 0, 0, tzinfo=datetime.timezone.utc)
    return int(dt.timestamp())


def _serialize_option_order(order: Order, spot: float) -> dict:
    parts    = order.instrument.split("-")
    is_call  = parts[-1] == "C"
    strike   = float(parts[-2])
    expiry_ts = _parse_expiry(parts[1])
    vol      = order.price or 0.01
    params: dict[str, Any] = {
        "instrument_name": order.instrument,
        "amount":          order.size,
        "type":            order.order_type,
        "price":           round(vol_to_price(vol, spot, strike, expiry_ts, is_call), 6),
        "label":           order.client_id,
    }
    if order.post_only:
        params["post_only"] = True
    if order.reduce_only:
        params["reduce_only"] = True
    return params


def _serialize_perp_order(order: Order) -> dict:
    params: dict[str, Any] = {
        "instrument_name": order.instrument,
        "amount":          order.size,
        "type":            order.order_type,
        "label":           order.client_id,
    }
    if order.price is not None and order.order_type == "limit":
        params["price"] = order.price
    if order.reduce_only:
        params["reduce_only"] = True
    return params


# ---- RPC client with reconnect ----------------------------------------------

@dataclass
class _PendingRequest:
    future:  asyncio.Future
    sent_ms: int
    method:  str


class JsonRpcClient:
    """
    JSON-RPC 2.0 over WebSocket with automatic reconnect.

    Public feed (market_data.py) already had reconnect - private WS didn't.
    Now both use the same pattern: exponential backoff, max_attempts cap,
    re-auth + re-subscribe after each reconnect.

    on_reconnect: async callback fired after a successful reconnect.
    Caller uses this to re-authenticate and re-subscribe.
    """

    def __init__(
        self,
        ws_url:            str,
        reconnect_delay_s: float,
        max_reconnect:     int = 10,
        on_reconnect:      Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._url              = ws_url
        self._reconnect_delay  = reconnect_delay_s
        self._max_reconnect    = max_reconnect
        self._on_reconnect     = on_reconnect
        self._ws               = None
        self._pending: dict[int, _PendingRequest] = {}
        self._id_counter       = 0
        self._connected        = asyncio.Event()
        self._subscriptions: dict[str, Callable[[dict], Awaitable[None]]] = {}
        self._recv_task: asyncio.Task | None = None
        self._reconnect_count  = 0
        self._running          = False

    async def connect(self) -> None:
        self._running = True
        await self._do_connect()
        self._recv_task = asyncio.create_task(self._recv_loop(), name="private_ws_recv")

    async def _do_connect(self) -> None:
        self._ws = await websockets.connect(
            self._url,
            ping_interval = 20,
            ping_timeout  = 40,
            max_size      = 2 ** 22,
        )
        self._connected.set()
        self._reconnect_count = 0
        log.info(f"private WS connected: {self._url}")

    async def disconnect(self) -> None:
        self._running = False
        if self._recv_task:
            self._recv_task.cancel()
        if self._ws:
            await self._ws.close()
        self._connected.clear()

    async def call(self, method: str, params: dict, timeout_s: float = 5.0) -> dict:
        await self._connected.wait()

        req_id = self._next_id()
        future = asyncio.get_event_loop().create_future()
        self._pending[req_id] = _PendingRequest(future=future, sent_ms=_now_ms(), method=method)

        msg = json.dumps({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        try:
            await self._ws.send(msg)
        except Exception as e:
            self._pending.pop(req_id, None)
            raise

        try:
            return await asyncio.wait_for(future, timeout=timeout_s)
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            raise asyncio.TimeoutError(f"{method} timed out after {timeout_s}s")

    def subscribe(self, channel: str, handler: Callable[[dict], Awaitable[None]]) -> None:
        self._subscriptions[channel] = handler

    async def _recv_loop(self) -> None:
        """Runs until disconnect(). Handles reconnect on any connection drop."""
        while self._running:
            try:
                raw = await self._ws.recv()
                await self._dispatch(json.loads(raw))
            except (ConnectionClosed, WebSocketException) as e:
                log.warning(f"private WS dropped: {e}")
                self._connected.clear()
                self._fail_pending(ConnectionError("WS dropped"))
                if not self._running:
                    break
                await self._reconnect_loop()
            except json.JSONDecodeError as e:
                log.error(f"bad JSON: {e}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"recv error: {e}", exc_info=True)

    async def _reconnect_loop(self) -> None:
        while self._running and self._reconnect_count < self._max_reconnect:
            delay = self._reconnect_delay * (2 ** self._reconnect_count)
            delay = min(delay, 60.0)
            self._reconnect_count += 1
            log.info(f"reconnecting private WS in {delay:.1f}s (attempt {self._reconnect_count})")
            await asyncio.sleep(delay)
            try:
                await self._do_connect()
                if self._on_reconnect:
                    await self._on_reconnect()
                log.info("private WS reconnected ok")
                return
            except Exception as e:
                log.error(f"reconnect attempt {self._reconnect_count} failed: {e}")

        if self._reconnect_count >= self._max_reconnect:
            log.error(f"private WS: max reconnect attempts reached ({self._max_reconnect})")
            # risk engine will catch WS silence via health tracker

    def _fail_pending(self, exc: Exception) -> None:
        for req in self._pending.values():
            if not req.future.done():
                req.future.set_exception(exc)
        self._pending.clear()

    async def _dispatch(self, msg: dict) -> None:
        if "id" in msg:
            req_id = msg["id"]
            req    = self._pending.pop(req_id, None)
            if req is None:
                return
            if "error" in msg:
                err = msg["error"]
                if not req.future.done():
                    req.future.set_exception(
                        RuntimeError(f"Deribit {err.get('code')}: {err.get('message')}")
                    )
            else:
                if not req.future.done():
                    req.future.set_result(msg.get("result", {}))
            return

        if msg.get("method") == "subscription":
            params  = msg.get("params", {})
            channel = params.get("channel", "")
            data    = params.get("data", {})
            handler = self._subscriptions.get(channel)
            if handler:
                try:
                    await handler(data)
                except Exception as e:
                    log.error(f"subscription handler [{channel}]: {e}")

    def _next_id(self) -> int:
        self._id_counter += 1
        return self._id_counter


# ---- auth -------------------------------------------------------------------

class DeribitAuth:
    def __init__(self, client_id: str, client_secret: str) -> None:
        self._id      = client_id
        self._secret  = client_secret
        self._token   = ""
        self._refresh = ""
        self._expires_at = 0.0

    async def authenticate(self, rpc: JsonRpcClient) -> None:
        result = await rpc.call("public/auth", {
            "grant_type":    "client_credentials",
            "client_id":     self._id,
            "client_secret": self._secret,
        })
        self._token      = result["access_token"]
        self._refresh    = result.get("refresh_token", "")
        self._expires_at = time.monotonic() + result.get("expires_in", 900) - 60
        log.info("gateway authenticated")

    async def refresh(self, rpc: JsonRpcClient) -> None:
        if not self._refresh:
            await self.authenticate(rpc)
            return
        try:
            result = await rpc.call("public/auth", {
                "grant_type":    "refresh_token",
                "refresh_token": self._refresh,
            })
            self._token      = result["access_token"]
            self._refresh    = result.get("refresh_token", self._refresh)
            self._expires_at = time.monotonic() + result.get("expires_in", 900) - 60
        except Exception as e:
            log.warning(f"token refresh failed, re-authenticating: {e}")
            await self.authenticate(rpc)

    def needs_refresh(self) -> bool:
        return time.monotonic() >= self._expires_at

    @property
    def token(self) -> str:
        return self._token


# ---- fill handler (push, not poll) ------------------------------------------

class FillHandler:
    """
    Processes fill notifications pushed via user.trades.* private channel.
    This replaces the polling loop in _wait_for_fill — fills arrive in <10ms
    from the exchange push vs ~50-500ms from polling get_order_state.

    The execution engine's _wait_for_fill still exists as a fallback for
    the rare case where a push notification is dropped, but in normal
    operation fills arrive here first.
    """

    def __init__(self, on_fill: Callable[[str, float, float, float], Awaitable[None]]) -> None:
        self._on_fill = on_fill
        # cache of recent fills: exchange_order_id -> (size, price, iv)
        # execution engine checks this before polling
        self._fill_cache: dict[str, tuple[float, float, float]] = {}

    async def handle(self, data: dict | list) -> None:
        trades = data if isinstance(data, list) else [data]
        for trade in trades:
            eid   = trade.get("order_id", "")
            size  = float(trade.get("amount", 0.0))
            price = float(trade.get("price", 0.0))
            iv    = float(trade.get("iv", 0.0)) / 100.0 if trade.get("iv") else 0.0

            log.info(f"fill push | eid={eid} size={size} price={price:.6f} iv={iv:.2%}")
            # store in cache first so execution engine can find it
            self._fill_cache[eid] = (size, price, iv)

            try:
                await self._on_fill(eid, size, price, iv)
            except Exception as e:
                log.error(f"on_fill callback: {e}")

    def pop_fill(self, exchange_id: str) -> tuple[float, float, float] | None:
        """Called by execution engine before falling back to polling."""
        return self._fill_cache.pop(exchange_id, None)


# ---- main gateway -----------------------------------------------------------

class DeribitGateway(ExchangeGateway):
    """
    Production Deribit gateway.

    Improvements over v1:
      - Private WS reconnect: exponential backoff, re-auth + re-subscribe
        after each reconnect. on_reconnect callback wired internally.
      - Push fills: FillHandler processes user.trades.* push notifications.
        Execution engine can check fill_cache before polling get_order_state.
      - get_account_summary: used by risk engine MarginMonitor.
    """

    def __init__(self, cfg: Config, client_id: str, client_secret: str, asset: str = "BTC") -> None:
        self._cfg   = cfg
        self._asset = asset
        self._auth  = DeribitAuth(client_id, client_secret)
        self._rpc   = JsonRpcClient(
            ws_url            = cfg.market.venue.ws_url,
            reconnect_delay_s = cfg.market.venue.rate_limits.websocket_reconnect_delay_s,
            max_reconnect     = 10,
            on_reconnect      = self._on_private_reconnect,
        )
        self._spot:         float        = 0.0
        self._fill_handler: FillHandler | None = None
        self._token_refresh_task: asyncio.Task | None = None
        self._private_channels: list[str] = []   # remembered for re-subscribe

    @classmethod
    def from_env(cls, cfg: Config, asset: str = "BTC") -> "DeribitGateway":
        client_id     = os.environ.get("DERIBIT_CLIENT_ID", "")
        client_secret = os.environ.get("DERIBIT_CLIENT_SECRET", "")
        if not client_id or not client_secret:
            raise EnvironmentError(
                "DERIBIT_CLIENT_ID and DERIBIT_CLIENT_SECRET must be set in environment"
            )
        return cls(cfg, client_id, client_secret, asset)

    async def connect(self) -> None:
        await self._rpc.connect()

    async def authenticate(self) -> None:
        await self._auth.authenticate(self._rpc)
        self._token_refresh_task = asyncio.create_task(
            self._token_refresh_loop(), name="token_refresh"
        )

    async def _on_private_reconnect(self) -> None:
        """Re-auth and re-subscribe after a WS drop. Called by JsonRpcClient."""
        log.info("private WS reconnected - re-authenticating...")
        try:
            await self._auth.authenticate(self._rpc)
            if self._private_channels:
                await self._rpc.call("private/subscribe", {
                    "channels":     self._private_channels,
                    "access_token": self._auth.token,
                })
                log.info(f"re-subscribed to {len(self._private_channels)} private channels")
        except Exception as e:
            log.error(f"post-reconnect setup failed: {e}")

    def set_fill_callback(self, on_fill: Callable[[str, float, float, float], Awaitable[None]]) -> None:
        self._fill_handler = FillHandler(on_fill)

    async def subscribe_private(self) -> None:
        if self._fill_handler is None:
            raise RuntimeError("call set_fill_callback() before subscribe_private()")

        ac = self._cfg.market.assets[self._asset]
        channels = [
            f"user.trades.{ac.perp_instrument}.raw",
            f"user.trades.any.{self._asset.lower()}.raw",   # catches all option fills
        ]
        for ch in channels:
            self._rpc.subscribe(ch, self._fill_handler.handle)

        await self._rpc.call("private/subscribe", {
            "channels":     channels,
            "access_token": self._auth.token,
        })
        self._private_channels = channels
        log.info(f"subscribed to private channels: {channels}")

    def update_spot(self, spot: float) -> None:
        self._spot = spot

    def check_fill_cache(self, exchange_id: str) -> tuple[float, float, float] | None:
        """
        Check if a fill arrived via push before falling back to polling.
        Returns (size, price, iv) or None.
        Called by execution engine in _wait_for_fill.
        """
        if self._fill_handler is None:
            return None
        return self._fill_handler.pop_fill(exchange_id)

    # ---- ExchangeGateway interface ------------------------------------------

    async def send_order(self, order: Order) -> dict:
        await self._maybe_refresh()

        is_option = "PERPETUAL" not in order.instrument
        method    = f"private/{'buy' if order.side == 'buy' else 'sell'}"

        if is_option:
            if self._spot <= 0:
                raise RuntimeError("spot not set - call update_spot() first")
            params = _serialize_option_order(order, self._spot)
        else:
            params = _serialize_perp_order(order)

        params["access_token"] = self._auth.token
        t0     = _now_ms()
        result = await self._rpc.call(method, params)
        log.debug(f"order acked | {order.client_id} | {_now_ms()-t0}ms")

        order_data = result.get("order", result)
        return {
            "id":          order_data.get("order_id", ""),
            "order_state": order_data.get("order_state", "open"),
            "price":       order_data.get("price", 0.0),
            "amount":      order_data.get("amount", 0.0),
        }

    async def cancel_order(self, exchange_id: str) -> dict:
        await self._maybe_refresh()
        await self._rpc.call("private/cancel", {
            "order_id":     exchange_id,
            "access_token": self._auth.token,
        })
        return {"result": "ok", "order_id": exchange_id}

    async def cancel_all(self, instrument: str | None = None) -> dict:
        await self._maybe_refresh()
        params: dict[str, Any] = {"access_token": self._auth.token}
        method = "private/cancel_all"
        if instrument:
            params["instrument_name"] = instrument
            method = "private/cancel_all_by_instrument"
        await self._rpc.call(method, params)
        log.info(f"cancel_all | instrument={instrument}")
        return {"result": "ok"}

    async def get_order(self, exchange_id: str) -> dict:
        # check push cache first - avoids a round-trip in the happy path
        if self._fill_handler:
            cached = self._fill_handler.pop_fill(exchange_id)
            if cached:
                size, price, iv = cached
                return {"order_state": "filled", "filled_amount": size, "average_price": price}

        await self._maybe_refresh()
        result = await self._rpc.call("private/get_order_state", {
            "order_id":     exchange_id,
            "access_token": self._auth.token,
        })
        return {
            "order_state":   result.get("order_state", "unknown"),
            "filled_amount": result.get("filled_amount", 0.0),
            "average_price": result.get("average_price", 0.0),
        }

    async def get_positions(self) -> dict:
        await self._maybe_refresh()
        result = await self._rpc.call("private/get_positions", {
            "currency":     self._asset,
            "kind":         "any",
            "access_token": self._auth.token,
        })
        return {"positions": result}

    async def get_account_summary(self) -> dict:
        """Used by risk engine MarginMonitor. Returns equity and margin data."""
        await self._maybe_refresh()
        result = await self._rpc.call("private/get_account_summary", {
            "currency":     self._asset,
            "extended":     True,
            "access_token": self._auth.token,
        })
        return {
            "equity":          result.get("equity", 0.0),
            "initial_margin":  result.get("initial_margin", 0.0),
            "maintenance_margin": result.get("maintenance_margin", 0.0),
            "available_funds": result.get("available_funds", 0.0),
        }

    async def disconnect(self) -> None:
        if self._token_refresh_task:
            self._token_refresh_task.cancel()
        await self._rpc.disconnect()

    async def _maybe_refresh(self) -> None:
        if self._auth.needs_refresh():
            await self._auth.refresh(self._rpc)

    async def _token_refresh_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(30)
                if self._auth.needs_refresh():
                    await self._auth.refresh(self._rpc)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"token refresh: {e}")


def _now_ms() -> int:
    return int(time.monotonic() * 1000)

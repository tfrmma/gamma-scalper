"""
infra/deribit_gateway.py

Concrete implementation of ExchangeGateway for Deribit.

Handles:
  - Authenticated WS connection (client_credentials)
  - Order send/cancel via JSON-RPC over WS
  - Private channel subscriptions (fills, account updates)
  - Black-76 vol->price conversion for option orders
  - Automatic token refresh

Credentials: set DERIBIT_CLIENT_ID and DERIBIT_CLIENT_SECRET in environment
or pass directly to DeribitGateway().

Does NOT handle market data - that's market_data.py.
Two separate WS connections: one public (market data), one private (orders).
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Awaitable

try:
    import websockets
    from websockets.exceptions import ConnectionClosed
except ImportError:
    raise ImportError("pip install websockets")

from config.loader import Config
from core.execution import ExchangeGateway, Order, OrderStatus

log = logging.getLogger("gateway")


# ---- Black-76 pricer --------------------------------------------------------
# needed to convert vol quotes to dollar prices when sending option orders
# Deribit accepts both vol and price, but price is unambiguous

def _black76_price(
    F:       float,   # forward price (use index for options)
    K:       float,   # strike
    T:       float,   # time to expiry in years
    sigma:   float,   # annualized vol (e.g. 0.65)
    r:       float,   # risk-free rate (use 0 for crypto)
    is_call: bool,
) -> float:
    """
    Black-76 option price.
    Returns price in underlying units (BTC). Multiply by index to get USD.
    """
    if T <= 0 or sigma <= 0:
        # intrinsic value only
        intrinsic = max(0.0, F - K) if is_call else max(0.0, K - F)
        return intrinsic / F   # normalize to underlying units

    sqrt_T = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    N = _norm_cdf
    discount = math.exp(-r * T)

    if is_call:
        return discount * (F * N(d1) - K * N(d2)) / F
    else:
        return discount * (K * N(-d2) - F * N(-d1)) / F


def _norm_cdf(x: float) -> float:
    """Abramowitz & Stegun approximation. Fast, accurate enough."""
    if x < 0:
        return 1.0 - _norm_cdf(-x)
    t = 1.0 / (1.0 + 0.2316419 * x)
    poly = t * (0.319381530
              + t * (-0.356563782
              + t * (1.781477937
              + t * (-1.821255978
              + t * 1.330274429))))
    return 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x) * poly


def vol_to_price(
    vol:        float,   # annualized IV
    spot:       float,   # current index price
    strike:     float,
    expiry_ts:  int,     # unix timestamp
    is_call:    bool,
) -> float:
    """Convert vol quote to Deribit price (in underlying units, normalized to index)."""
    T = max(0.0, (expiry_ts - time.time()) / (365 * 24 * 3600))
    raw = _black76_price(F=spot, K=strike, T=T, sigma=vol, r=0.0, is_call=is_call)
    return max(raw, 0.0001)   # Deribit minimum


# ---- RPC infrastructure -----------------------------------------------------

@dataclass
class PendingRequest:
    future:     asyncio.Future
    sent_ms:    int
    method:     str


class JsonRpcClient:
    """
    JSON-RPC 2.0 over WebSocket.
    Tracks in-flight requests by id, resolves futures on response.
    Not sophisticated - it doesn't need to be for the request rates we're sending.
    """

    def __init__(self, ws_url: str, reconnect_delay_s: float) -> None:
        self._url              = ws_url
        self._reconnect_delay  = reconnect_delay_s
        self._ws               = None
        self._pending: dict[int, PendingRequest] = {}
        self._id_counter       = 0
        self._connected        = asyncio.Event()
        self._running          = False
        self._recv_task        = None

        # callbacks for unsolicited messages (fills, account updates)
        self._subscriptions: dict[str, Callable[[dict], Awaitable[None]]] = {}

    async def connect(self) -> None:
        self._ws = await websockets.connect(
            self._url,
            ping_interval=20,
            ping_timeout=40,
            max_size=2**22,
        )
        self._connected.set()
        self._recv_task = asyncio.create_task(self._recv_loop())
        log.info(f"gateway WS connected: {self._url}")

    async def disconnect(self) -> None:
        self._running = False
        if self._recv_task:
            self._recv_task.cancel()
        if self._ws:
            await self._ws.close()
        self._connected.clear()

    async def call(self, method: str, params: dict, timeout_s: float = 5.0) -> dict:
        """
        Send an RPC call and wait for the response.
        Raises asyncio.TimeoutError if exchange doesn't respond in time.
        Raises RuntimeError if exchange returns an error response.
        """
        await self._connected.wait()

        req_id  = self._next_id()
        future  = asyncio.get_event_loop().create_future()
        self._pending[req_id] = PendingRequest(future=future, sent_ms=_now_ms(), method=method)

        msg = json.dumps({
            "jsonrpc": "2.0",
            "id":      req_id,
            "method":  method,
            "params":  params,
        })

        try:
            await self._ws.send(msg)
        except Exception as e:
            self._pending.pop(req_id, None)
            raise

        try:
            result = await asyncio.wait_for(future, timeout=timeout_s)
            return result
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            raise asyncio.TimeoutError(f"{method} timed out after {timeout_s}s")

    def subscribe(self, channel: str, handler: Callable[[dict], Awaitable[None]]) -> None:
        self._subscriptions[channel] = handler

    async def _recv_loop(self) -> None:
        while True:
            try:
                raw = await self._ws.recv()
                await self._dispatch(json.loads(raw))
            except ConnectionClosed as e:
                log.warning(f"gateway WS closed: {e}")
                self._connected.clear()
                # fail all pending requests
                for req in self._pending.values():
                    if not req.future.done():
                        req.future.set_exception(ConnectionError("WS closed"))
                self._pending.clear()
                break
            except json.JSONDecodeError as e:
                log.error(f"bad JSON from gateway: {e}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"recv loop error: {e}", exc_info=True)

    async def _dispatch(self, msg: dict) -> None:
        # RPC response
        if "id" in msg:
            req_id = msg["id"]
            req    = self._pending.pop(req_id, None)
            if req is None:
                return
            if "error" in msg:
                err = msg["error"]
                if not req.future.done():
                    req.future.set_exception(
                        RuntimeError(f"Deribit error {err.get('code')}: {err.get('message')}")
                    )
            else:
                if not req.future.done():
                    req.future.set_result(msg.get("result", {}))
            return

        # subscription notification
        if msg.get("method") == "subscription":
            params  = msg.get("params", {})
            channel = params.get("channel", "")
            data    = params.get("data", {})
            handler = self._subscriptions.get(channel)
            if handler:
                try:
                    await handler(data)
                except Exception as e:
                    log.error(f"subscription handler error [{channel}]: {e}")

    def _next_id(self) -> int:
        self._id_counter += 1
        return self._id_counter


# ---- auth -------------------------------------------------------------------

class DeribitAuth:
    """
    client_credentials flow.
    Token expires after ~15 minutes - refresh before that.
    """

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
        expires_in       = result.get("expires_in", 900)
        self._expires_at = time.monotonic() + expires_in - 60  # refresh 1min early
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
            expires_in       = result.get("expires_in", 900)
            self._expires_at = time.monotonic() + expires_in - 60
            log.debug("gateway token refreshed")
        except Exception as e:
            log.warning(f"token refresh failed, re-authenticating: {e}")
            await self.authenticate(rpc)

    def needs_refresh(self) -> bool:
        return time.monotonic() >= self._expires_at

    @property
    def token(self) -> str:
        return self._token


# ---- order serialization ----------------------------------------------------

def _serialize_option_order(order: Order, spot: float) -> dict:
    """
    Convert our internal Order to Deribit private/buy or private/sell params.
    option order.price is in vol (e.g. 0.65) - convert to USD price via B76.
    Deribit wants price in index units (BTC amount per contract).
    """
    # parse strike and expiry from instrument name: "BTC-27JUN25-60000-C"
    parts    = order.instrument.split("-")
    is_call  = parts[-1] == "C"
    strike   = float(parts[-2])
    expiry_ts = _parse_expiry(parts[1])

    vol    = order.price or 0.01   # fallback shouldn't happen
    dollar_price = vol_to_price(vol, spot, strike, expiry_ts, is_call)

    params: dict[str, Any] = {
        "instrument_name": order.instrument,
        "amount":          order.size,
        "type":            order.order_type,
        "price":           round(dollar_price, 6),
        "label":           order.client_id,
    }
    if order.post_only:
        params["post_only"] = True
    if order.reduce_only:
        params["reduce_only"] = True

    return params


def _serialize_perp_order(order: Order) -> dict:
    """
    Perp order. price is a dollar price, size is USD notional.
    Deribit perp uses USD contracts - amount = notional directly.
    """
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


def _parse_expiry(s: str) -> int:
    """Parse "27JUN25" -> unix timestamp at 08:00 UTC."""
    import datetime
    month_map = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
        "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
        "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }
    day   = int(s[:2])
    month = month_map[s[2:5]]
    year  = 2000 + int(s[5:])
    dt    = datetime.datetime(year, month, day, 8, 0, 0, tzinfo=datetime.timezone.utc)
    return int(dt.timestamp())


# ---- fill notification handler ----------------------------------------------

class FillHandler:
    """
    Processes fill notifications from the private user.trades channel.
    Calls back into the execution engine and state engine on each fill.

    on_fill: async callback(exchange_id, size, price, iv)
    """

    def __init__(self, on_fill: Callable[[str, float, float, float], Awaitable[None]]) -> None:
        self._on_fill = on_fill

    async def handle(self, data: dict | list) -> None:
        trades = data if isinstance(data, list) else [data]
        for trade in trades:
            eid   = trade.get("order_id", "")
            size  = trade.get("amount", 0.0)
            price = trade.get("price", 0.0)
            iv    = trade.get("iv", 0.0) / 100.0 if trade.get("iv") else 0.0
            log.info(f"fill | eid={eid} size={size} price={price:.6f} iv={iv:.2%}")
            try:
                await self._on_fill(eid, float(size), float(price), iv)
            except Exception as e:
                log.error(f"on_fill callback error: {e}")


# ---- main gateway -----------------------------------------------------------

class DeribitGateway(ExchangeGateway):
    """
    Production Deribit gateway.

    Usage:
        gw = DeribitGateway.from_env(cfg)
        await gw.connect()
        await gw.authenticate()
        # subscribe to fills
        gw.on_fill = my_fill_handler
        # then pass to ExecutionEngine
    """

    def __init__(
        self,
        cfg:           Config,
        client_id:     str,
        client_secret: str,
        asset:         str = "BTC",
    ) -> None:
        self._cfg    = cfg
        self._asset  = asset
        self._auth   = DeribitAuth(client_id, client_secret)
        self._rpc    = JsonRpcClient(
            ws_url            = cfg.market.venue.ws_url,
            reconnect_delay_s = cfg.market.venue.rate_limits.websocket_reconnect_delay_s,
        )

        self._spot:     float = 0.0   # updated from state engine before order sends
        self._on_fill:  Callable | None = None
        self._fill_handler: FillHandler | None = None
        self._token_refresh_task: asyncio.Task | None = None

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
        self._token_refresh_task = asyncio.create_task(self._token_refresh_loop())

    def set_fill_callback(
        self,
        on_fill: Callable[[str, float, float, float], Awaitable[None]],
    ) -> None:
        """Set callback for fill notifications. Call before subscribe_private."""
        self._on_fill     = on_fill
        self._fill_handler = FillHandler(on_fill)

    async def subscribe_private(self) -> None:
        """Subscribe to user.trades for fill notifications."""
        if self._fill_handler is None:
            raise RuntimeError("set_fill_callback() before subscribe_private()")

        channel = f"user.trades.{self._cfg.market.assets[self._asset].perp_instrument}.raw"
        self._rpc.subscribe(channel, self._fill_handler.handle)

        # also subscribe to option fills - all at once
        await self._rpc.call("private/subscribe", {
            "channels": [channel],
            "access_token": self._auth.token,
        })
        log.info(f"subscribed to private fills | channel={channel}")

    def update_spot(self, spot: float) -> None:
        """Called by main loop to keep spot price current for vol->price conversion."""
        self._spot = spot

    # ---- ExchangeGateway interface ------------------------------------------

    async def send_order(self, order: Order) -> dict:
        if self._auth.needs_refresh():
            await self._auth.refresh(self._rpc)

        is_option = "PERPETUAL" not in order.instrument
        method    = f"private/{'buy' if order.side == 'buy' else 'sell'}"

        if is_option:
            if self._spot <= 0:
                raise RuntimeError("spot price not set - call update_spot() before sending option orders")
            params = _serialize_option_order(order, self._spot)
        else:
            params = _serialize_perp_order(order)

        params["access_token"] = self._auth.token

        t0     = _now_ms()
        result = await self._rpc.call(method, params)
        elapsed = _now_ms() - t0

        log.debug(f"order acked | {order.client_id} | latency={elapsed}ms")

        # Deribit returns {"order": {...}, "trades": [...]}
        order_data = result.get("order", result)
        return {
            "id":          order_data.get("order_id", ""),
            "order_state": order_data.get("order_state", "open"),
            "price":       order_data.get("price", 0.0),
            "amount":      order_data.get("amount", 0.0),
        }

    async def cancel_order(self, exchange_id: str) -> dict:
        if self._auth.needs_refresh():
            await self._auth.refresh(self._rpc)

        result = await self._rpc.call("private/cancel", {
            "order_id":     exchange_id,
            "access_token": self._auth.token,
        })
        return {"result": "ok", "order_id": exchange_id}

    async def cancel_all(self, instrument: str | None = None) -> dict:
        if self._auth.needs_refresh():
            await self._auth.refresh(self._rpc)

        params: dict[str, Any] = {"access_token": self._auth.token}
        method = "private/cancel_all"

        if instrument:
            params["instrument_name"] = instrument
            method = "private/cancel_all_by_instrument"

        result = await self._rpc.call(method, params)
        log.info(f"cancel_all | instrument={instrument} result={result}")
        return {"result": "ok"}

    async def get_order(self, exchange_id: str) -> dict:
        if self._auth.needs_refresh():
            await self._auth.refresh(self._rpc)

        result = await self._rpc.call("private/get_order_state", {
            "order_id":     exchange_id,
            "access_token": self._auth.token,
        })
        return {
            "order_state":    result.get("order_state", "unknown"),
            "filled_amount":  result.get("filled_amount", 0.0),
            "average_price":  result.get("average_price", 0.0),
        }

    async def get_positions(self) -> dict:
        if self._auth.needs_refresh():
            await self._auth.refresh(self._rpc)

        result = await self._rpc.call("private/get_positions", {
            "currency":     self._asset,
            "kind":         "any",
            "access_token": self._auth.token,
        })
        return {"positions": result}

    async def disconnect(self) -> None:
        if self._token_refresh_task:
            self._token_refresh_task.cancel()
        await self._rpc.disconnect()

    # ---- token refresh loop -------------------------------------------------

    async def _token_refresh_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(30)  # check every 30s
                if self._auth.needs_refresh():
                    await self._auth.refresh(self._rpc)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"token refresh loop error: {e}")


# ---- util -------------------------------------------------------------------

def _now_ms() -> int:
    return int(time.monotonic() * 1000)

"""
WebSocket live data feed for Angel One SmartAPI.

Subscribes to real-time tick data via SmartWebSocketV2 and builds
3-minute OHLCV candles locally, eliminating REST API polling during
live trading.

Usage:
    feed = WebSocketFeed(client, STOCKS)
    feed.start()
    # ... in scan loop ...
    bars = feed.get_completed_bars("BAJAJ-AUTO")
    # ... when done ...
    feed.stop()
"""

import logging
import threading
import time as _time
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from fetch_data import AngelOneClient

CANDLE_SECONDS = 180
MKT_OPEN_H, MKT_OPEN_M = 9, 15

flog = logging.getLogger("live_signals")


@dataclass
class _RunningCandle:
    """Accumulator for a single in-progress 3-minute bar."""
    open_ts: datetime
    open: float = 0.0
    high: float = float("-inf")
    low: float = float("inf")
    close: float = 0.0
    cum_vol_start: int = 0
    cum_vol_latest: int = 0
    tick_count: int = 0

    @property
    def volume(self) -> int:
        return max(0, self.cum_vol_latest - self.cum_vol_start)

    def update(self, ltp: float, cum_vol: int):
        if self.tick_count == 0:
            self.open = ltp
            self.cum_vol_start = cum_vol
        self.high = max(self.high, ltp)
        self.low = min(self.low, ltp)
        self.close = ltp
        self.cum_vol_latest = cum_vol
        self.tick_count += 1

    def to_dict(self) -> dict:
        return {
            "datetime": self.open_ts,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


class WebSocketFeed:
    """Streams live ticks via SmartWebSocketV2 and builds 3-min candles.

    Thread-safe: WebSocket callbacks run on a daemon thread; the main
    thread reads completed candles via get_completed_bars().
    """

    def __init__(self, client: AngelOneClient, stocks: list[tuple[str, str]]):
        self._client = client
        self._stocks = stocks

        self._token_to_sym: dict[str, str] = {}
        self._sym_tokens: list[str] = []
        for sym, _ in stocks:
            token = AngelOneClient.SYMBOL_TOKENS.get(sym)
            if token:
                self._token_to_sym[token] = sym
                self._sym_tokens.append(token)

        self._lock = threading.Lock()
        self._running: dict[str, _RunningCandle] = {}
        self._completed: dict[str, list[dict]] = {s: [] for s, _ in stocks}
        self._last_cum_vol: dict[str, int] = {}
        self._last_finalized_open: dict[str, datetime] = {}

        self._sws = None
        self._connected = False
        self._last_tick_time: float = 0.0
        self._reconnect_count = 0
        self._should_run = False
        self._ws_thread: threading.Thread | None = None
        self._outage_recovered = False
        self._disconnect_time: float = 0.0

    # ── Public API ────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def seconds_since_last_tick(self) -> float:
        if self._last_tick_time == 0:
            return float("inf")
        return _time.monotonic() - self._last_tick_time

    @property
    def had_outage(self) -> bool:
        """True if WebSocket recovered from a disconnect since last clear."""
        return self._outage_recovered

    def clear_outage(self):
        """Acknowledge the outage — called after scan loop does REST gap-fill."""
        self._outage_recovered = False

    def start(self):
        """Start WebSocket feed on a daemon thread."""
        if not self._client.jwt_token or not self._client.feed_token:
            raise RuntimeError(
                "Client must be connected first (call client.connect())"
            )
        self._should_run = True
        self._ws_thread = threading.Thread(
            target=self._run_loop, daemon=True, name="ws-feed",
        )
        self._ws_thread.start()

        for _ in range(20):
            if self._connected:
                break
            _time.sleep(0.5)

        if self._connected:
            flog.info("WS feed started — %d tokens subscribed", len(self._sym_tokens))
        else:
            flog.warning("WS not connected after 10s — retrying in background")

    def stop(self):
        """Stop the WebSocket feed and its thread."""
        self._should_run = False
        if self._sws:
            try:
                self._sws.close_connection()
            except Exception:
                pass
        self._connected = False
        flog.info("WS feed stopped")

    def flush_elapsed_bars(self, as_of: datetime | None = None) -> int:
        """Finalize any running candles whose 3-minute window has elapsed.

        This prevents a quiet symbol from surfacing a completed bar one full
        scan late just because its next tick has not arrived yet.
        """
        as_of = as_of or datetime.now()
        boundary = self._candle_boundary(as_of)
        flushed = 0

        with self._lock:
            for sym, current in list(self._running.items()):
                if boundary <= current.open_ts:
                    continue
                if current.tick_count > 0:
                    self._finalize_candle_locked(sym, current)
                    flushed += 1
                del self._running[sym]

        return flushed

    def get_completed_bars(self, sym: str) -> pd.DataFrame:
        """Drain and return all completed candle bars since the last call.

        Returns an empty DataFrame if no new bars are available.
        """
        with self._lock:
            bars = self._completed.get(sym, [])
            if not bars:
                return pd.DataFrame(
                    columns=["open", "high", "low", "close", "volume"]
                ).rename_axis("datetime")
            self._completed[sym] = []

        df = pd.DataFrame(bars)
        df = df.set_index("datetime").astype(
            {"open": float, "high": float, "low": float, "close": float, "volume": int}
        )
        return df.sort_index()

    # ── Candle Construction ───────────────────────────────────────────

    @staticmethod
    def _candle_boundary(dt: datetime) -> datetime:
        """Align a timestamp to its 3-minute candle open."""
        base = dt.replace(
            hour=MKT_OPEN_H, minute=MKT_OPEN_M, second=0, microsecond=0,
        )
        elapsed = (dt - base).total_seconds()
        if elapsed < 0:
            return base
        idx = int(elapsed / CANDLE_SECONDS)
        return base + timedelta(seconds=idx * CANDLE_SECONDS)

    def _finalize_candle_locked(self, sym: str, candle: _RunningCandle) -> None:
        """Append a finished candle and remember its open timestamp."""
        self._completed[sym].append(candle.to_dict())
        self._last_finalized_open[sym] = candle.open_ts

    def _process_tick(self, message: dict):
        token = str(message.get("token", ""))
        sym = self._token_to_sym.get(token)
        if not sym:
            return

        ltp_raw = message.get("last_traded_price", 0)
        ltp = ltp_raw / 100.0
        if ltp <= 0:
            return

        cum_vol = message.get("volume_trade_for_the_day", 0)

        ts_ms = message.get("exchange_timestamp")
        tick_dt = datetime.fromtimestamp(ts_ms / 1000) if ts_ms else datetime.now()

        candle_open = self._candle_boundary(tick_dt)
        self._last_tick_time = _time.monotonic()

        with self._lock:
            current = self._running.get(sym)
            last_finalized_open = self._last_finalized_open.get(sym)

            # Ignore late ticks for a candle we have already finalized.
            if last_finalized_open is not None and candle_open <= last_finalized_open:
                return

            if current is not None and candle_open > current.open_ts:
                if current.tick_count > 0:
                    self._finalize_candle_locked(sym, current)
                current = None

            if current is None:
                vol_base = self._last_cum_vol.get(sym, cum_vol)
                current = _RunningCandle(
                    open_ts=candle_open, cum_vol_start=vol_base,
                )
                self._running[sym] = current

            current.update(ltp, cum_vol)
            self._last_cum_vol[sym] = cum_vol

    # ── WebSocket Callbacks ───────────────────────────────────────────

    def _on_data(self, wsapp, message):
        try:
            self._process_tick(message)
        except Exception as e:
            flog.warning("WS tick error: %s", e)

    def _on_open(self, wsapp):
        was_disconnected = self._disconnect_time > 0
        self._connected = True
        self._reconnect_count = 0

        if was_disconnected:
            gap = _time.monotonic() - self._disconnect_time
            flog.warning("WS recovered after %.0fs outage — clearing stale candle data", gap)
            with self._lock:
                self._running.clear()
                self._last_cum_vol.clear()
            self._outage_recovered = True
            self._disconnect_time = 0.0

        flog.info("WS connected — subscribing %d tokens (mode=2 Quote)",
                  len(self._sym_tokens))
        token_list = [{"exchangeType": 1, "tokens": list(self._sym_tokens)}]
        self._sws.subscribe("live_feed", 2, token_list)

    def _on_error(self, wsapp, error):
        flog.warning("WS error: %s", error)

    def _on_close(self, wsapp):
        self._connected = False
        if self._disconnect_time == 0.0 and self._should_run:
            self._disconnect_time = _time.monotonic()
        flog.warning("WS disconnected")

    # ── Connection Management ─────────────────────────────────────────

    def _create_and_connect(self):
        """Create a SmartWebSocketV2 instance and block until it disconnects."""
        import logging as _logging
        try:
            import logzero as _lz
            _lz.loglevel(_logging.CRITICAL)
            _lz.logfile(None)
        except ImportError:
            pass
        for name in ("smartConnect", "SmartApi", "logzero",
                      "SmartWebSocketV2", "websocket", "urllib3"):
            _lg = _logging.getLogger(name)
            _lg.setLevel(_logging.CRITICAL)
            _lg.handlers = []
            _lg.addHandler(_logging.NullHandler())

        from SmartApi.smartWebSocketV2 import SmartWebSocketV2

        self._sws = SmartWebSocketV2(
            self._client.jwt_token,
            self._client.api_key,
            self._client.client_id,
            self._client.feed_token,
        )
        self._sws.on_open = self._on_open
        self._sws.on_data = self._on_data
        self._sws.on_error = self._on_error
        self._sws.on_close = self._on_close
        self._sws.connect()

    def _run_loop(self):
        """Auto-reconnecting WebSocket loop. Runs on the daemon thread."""
        while self._should_run:
            try:
                self._create_and_connect()
            except Exception as e:
                flog.warning("WS connection error: %s", e)

            self._connected = False
            if not self._should_run:
                break

            delay = min(2 ** self._reconnect_count, 30)
            self._reconnect_count += 1
            flog.info("WS reconnecting in %ds (attempt %d)...",
                      delay, self._reconnect_count)
            _time.sleep(delay)

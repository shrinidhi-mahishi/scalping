"""
Angel One SmartAPI data fetcher and sample data generator.

Provides two data sources:
  1. Live historical OHLCV from Angel One (requires API credentials)
  2. Synthetic sample data for offline backtesting
"""

import os
import time as _time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


# ─── Angel One Client ────────────────────────────────────────────────────────


class AngelOneClient:
    """Fetches intraday OHLCV candles via Angel One SmartAPI."""

    SYMBOL_TOKENS = {
        "ADANIENT": "25",
        "ADANIPORTS": "15083",
        "APOLLOHOSP": "157",
        "ASIANPAINT": "236",
        "AXISBANK": "5900",
        "BAJAJ-AUTO": "16669",
        "BAJAJFINSV": "16675",
        "BAJFINANCE": "317",
        "BEL": "383",
        "BHARTIARTL": "10604",
        "CIPLA": "694",
        "COALINDIA": "20374",
        "DRREDDY": "881",
        "EICHERMOT": "910",
        "GRASIM": "1232",
        "HCLTECH": "7229",
        "HDFCBANK": "1333",
        "HDFCLIFE": "467",
        "HINDALCO": "1363",
        "HINDUNILVR": "1394",
        "ICICIBANK": "4963",
        "INDIGO": "11195",
        "INFY": "1594",
        "ITC": "1660",
        "JIOFIN": "18143",
        "JSWSTEEL": "11723",
        "KOTAKBANK": "1922",
        "LT": "11483",
        "M&M": "2031",
        "MARUTI": "10999",
        "NESTLEIND": "17963",
        "NTPC": "11630",
        "ONGC": "2475",
        "POWERGRID": "14977",
        "RELIANCE": "2885",
        "SBILIFE": "21808",
        "SBIN": "3045",
        "SHRIRAMFIN": "4306",
        "SUNPHARMA": "3351",
        "TATACONSUM": "3432",
        "TATAMOTORS": "3456",
        "TATASTEEL": "3499",
        "TCS": "11536",
        "TECHM": "13538",
        "TITAN": "3506",
        "TRENT": "1964",
        "ULTRACEMCO": "11532",
        "WIPRO": "3787",
    }

    def __init__(self):
        self.api_key = os.getenv("ANGEL_API_KEY")
        self.client_id = os.getenv("ANGEL_CLIENT_ID")
        self.password = os.getenv("ANGEL_PASSWORD")
        self.totp_secret = os.getenv("ANGEL_TOTP_SECRET")
        self._conn = None

    def connect(self):
        from SmartApi import SmartConnect
        import pyotp

        self._conn = SmartConnect(api_key=self.api_key)
        totp = pyotp.TOTP(self.totp_secret).now()
        resp = self._conn.generateSession(self.client_id, self.password, totp)

        if not resp.get("status"):
            raise ConnectionError(f"Angel One login failed: {resp.get('message')}")
        print(f"[Angel One] Logged in as {self.client_id}")
        return self

    def fetch_candles(
        self,
        symbol: str,
        from_date: str,
        to_date: str,
        interval: str = "THREE_MINUTE",
        exchange: str = "NSE",
    ) -> pd.DataFrame:
        token = self.SYMBOL_TOKENS.get(symbol.upper())
        if not token:
            raise ValueError(
                f"Unknown symbol '{symbol}'. Known: {list(self.SYMBOL_TOKENS)}"
            )

        resp = self._conn.getCandleData(
            {
                "exchange": exchange,
                "symboltoken": token,
                "interval": interval,
                "fromdate": from_date,
                "todate": to_date,
            }
        )

        if not resp.get("status") or not resp.get("data"):
            raise RuntimeError(f"Candle fetch failed: {resp.get('message', 'empty')}")

        df = pd.DataFrame(
            resp["data"],
            columns=["datetime", "open", "high", "low", "close", "volume"],
        )
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df["datetime"] = df["datetime"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        df = df.set_index("datetime").astype(
            {"open": float, "high": float, "low": float, "close": float, "volume": int}
        )
        return df

    def fetch_history(
        self,
        symbol: str,
        days: int = 30,
        interval: str = "THREE_MINUTE",
    ) -> pd.DataFrame:
        """Fetch multi-day intraday data (chunked to respect API's 5-day limit)."""
        if not self._conn:
            self.connect()

        chunks: list[pd.DataFrame] = []
        end = datetime.now()
        remaining = days

        while remaining > 0:
            span = min(5, remaining)
            start = end - timedelta(days=span)
            try:
                df = self.fetch_candles(
                    symbol,
                    start.strftime("%Y-%m-%d 09:15"),
                    end.strftime("%Y-%m-%d 15:30"),
                    interval,
                )
                chunks.append(df)
            except Exception as e:
                print(f"[warn] {start.date()} → {end.date()}: {e}")
            end = start
            remaining -= span
            _time.sleep(0.5)

        if not chunks:
            raise RuntimeError("No data fetched from Angel One")

        combined = pd.concat(chunks).sort_index()
        return combined[~combined.index.duplicated(keep="first")]


# ─── Sample Data Generator ───────────────────────────────────────────────────


def generate_sample_data(
    symbol: str = "RELIANCE",
    days: int = 30,
    base_price: float = 2500.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic 3-minute OHLCV data with:
      - U-shaped volume profile (high at open/close, low midday)
      - Intraday mean-reversion + momentum
      - Overnight gaps
    """
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    price = base_price

    trade_dates = pd.bdate_range(end=pd.Timestamp("2026-02-20"), periods=days)

    for date in trade_dates:
        day_open = price * (1 + rng.normal(0, 0.004))
        price = day_open
        intraday_drift = rng.normal(0, 0.00012)

        t = pd.Timestamp(date.year, date.month, date.day, 9, 15)
        end_t = pd.Timestamp(date.year, date.month, date.day, 15, 30)
        bar_idx = 0

        while t <= end_t:
            hf = t.hour + t.minute / 60.0

            if hf < 10.0:
                vol_mult = 2.5
            elif hf > 14.5:
                vol_mult = 2.0
            elif 12.0 < hf < 13.5:
                vol_mult = 0.4
            else:
                vol_mult = 1.0

            volume = max(100, int(rng.lognormal(11, 0.5) * vol_mult))

            momentum = intraday_drift * bar_idx
            mean_rev = (day_open - price) * 0.0008
            shock = rng.normal(momentum + mean_rev, price * 0.0012)

            o = round(price, 2)
            c = round(price + shock, 2)
            spread = abs(rng.normal(0, price * 0.0004))
            h = round(max(o, c) + spread, 2)
            l = round(min(o, c) - spread, 2)

            records.append(
                {
                    "datetime": t,
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "volume": volume,
                }
            )
            price = c
            t += pd.Timedelta(minutes=3)
            bar_idx += 1

    df = pd.DataFrame(records).set_index("datetime")
    df.attrs["symbol"] = symbol
    return df


# ─── CSV helpers ──────────────────────────────────────────────────────────────


def save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path)
    print(f"[data] Saved {len(df)} rows → {path}")


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="datetime", parse_dates=True)
    if df.index.tz is not None:
        df.index = df.index.tz_convert("Asia/Kolkata").tz_localize(None)
    return df

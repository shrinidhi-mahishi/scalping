"""
Live Signal Generator for NSE Intraday Scalping Strategy

Polls Angel One SmartAPI every 3 minutes (at HH:MM:02) for 9 stocks,
computes indicators in real-time (pandas/numpy — no backtrader), and
sends BUY/SHORT alerts to console + Telegram.

Daily Lifecycle:
  Phase 1  09:14  Warm-Up   — connect, fetch daily candles (regime EMA),
                              fetch 2 days of 3-min data (indicator warmup)
  Phase 2  09:15  Silent    — track indicators every 3 min, NO signals
  Phase 3  10:00  Active AM — morning trading window, signals enabled
  Phase 4  12:00  Midday    — keep tracking indicators, signals blocked
  Phase 5  14:00  Active PM — afternoon trading window, signals enabled
  Phase 6  15:00  Shutdown  — stop fetching, let bracket orders settle

Filters : Daily 21-EMA regime gate + optional 15-min 21-EMA HTF per stock
          VWAP extension limit ±1.5× ATR (surgical: HDFCLIFE, TITAN only)
Strategy: EMA(9/21) crossover · VWAP · RSI(9) · Volume SMA(20) · ATR(14)
RR      : 1 : 1.5  (SL = 1×ATR, TP = 1.5×ATR)
Sizing  : 1% risk per trade, 5× leverage cap
Exit    : Stagnation exit if profit < 0.2×ATR after 8 bars (skip: HDFCLIFE, TITAN, ONGC)

Usage:
    python live_signals.py                  # default ₹50,000 capital
    python live_signals.py --capital 100000 # custom capital
    python live_signals.py --dry-run        # scan last session, no live loop
"""

import argparse
import csv
import logging
import os
import sys
import time as _time
from datetime import datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from fetch_data import AngelOneClient, load_csv

load_dotenv()


# ─── Stock Configuration ──────────────────────────────────────────────────

STOCKS = [
    ("HDFCLIFE", "Finance"),
    ("TATASTEEL", "Metals"),
    ("TITAN", "Consumer"),
    ("M&M", "Auto"),
    ("ADANIPORTS", "Conglomerate"),
    ("ONGC", "Energy"),
    ("BAJAJFINSV", "Finance"),
    ("HINDALCO", "Metals"),
    ("SBILIFE", "Finance"),
]

HTF_STOCKS = {"TATASTEEL", "M&M", "ONGC", "BAJAJFINSV", "HINDALCO", "SBILIFE"}

STAGNATION_SKIP = {"HDFCLIFE", "TITAN", "ONGC"}
STAG_MAX_BARS = 8
STAG_MIN_PROFIT_ATR = 0.2

VWAP_EXT_STOCKS = {"HDFCLIFE", "TITAN"}
VWAP_EXT_ATR = 1.5


# ─── Strategy Parameters ──────────────────────────────────────────────────

FAST_EMA = 9
SLOW_EMA = 21
RSI_PERIOD = 9
VOL_SMA = 20
ATR_PERIOD = 14
HTF_EMA_PERIOD = 21
HTF_MIN_BARS_15M = 40
REGIME_EMA_PERIOD = 21

RR = 1.5
RISK_PCT = 0.01
LEV_CAP = 5.0

LONG_RSI_RANGE = (40, 70)
SHORT_RSI_RANGE = (30, 60)


# ─── Time Rules & Phases ──────────────────────────────────────────────────

MKT_OPEN = time(9, 15)
ENTRY_AM = (time(10, 0), time(12, 0))
ENTRY_PM = (time(14, 0), time(15, 0))
MKT_CLOSE = time(15, 0)

COOLDOWN = timedelta(minutes=15)
MAX_SIGNALS_PER_STOCK = 2
WARMUP_3MIN_DAYS = 5
REGIME_FETCH_DAYS = 60
POLL_BUFFER_SEC = 2

DATA_DIR = Path(__file__).parent / "data"
LOG_DIR = Path(__file__).parent / "logs"


def _init_file_logger() -> logging.Logger:
    LOG_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("live_signals")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        fh = logging.FileHandler(
            LOG_DIR / f"live_{datetime.now():%Y-%m-%d}.log", encoding="utf-8",
        )
        fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(fh)
    return logger


flog = _init_file_logger()

_LOG_COLUMNS = [
    "date", "time", "symbol", "sector", "direction", "price", "sl", "tp",
    "atr", "qty", "risk", "vwap", "rsi", "vol_ratio", "htf_ema", "regime_ema",
    "regime_trend", "phase",
]

PHASES = {
    "PRE_MARKET": "Pre-market",
    "SILENT":     "Phase 2 · Silent tracking",
    "ACTIVE_AM":  "Phase 3 · Morning window",
    "MIDDAY":     "Phase 4 · Midday sleep",
    "ACTIVE_PM":  "Phase 5 · Afternoon window",
    "SHUTDOWN":   "Phase 6 · Shutdown",
}


def get_phase(t: time) -> str:
    if t < MKT_OPEN:
        return "PRE_MARKET"
    if t < ENTRY_AM[0]:
        return "SILENT"
    if t <= ENTRY_AM[1]:
        return "ACTIVE_AM"
    if t < ENTRY_PM[0]:
        return "MIDDAY"
    if t <= ENTRY_PM[1]:
        return "ACTIVE_PM"
    return "SHUTDOWN"


# ─── Terminal Colors ───────────────────────────────────────────────────────

GRN, RED, YLW, CYN = "\033[92m", "\033[91m", "\033[93m", "\033[96m"
BLD, DIM, RST = "\033[1m", "\033[2m", "\033[0m"


# ─── Telegram ─────────────────────────────────────────────────────────────

TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")

_TELEGRAM_SETUP = """\
  To enable Telegram alerts, add these to your .env file:
    TELEGRAM_BOT_TOKEN=<your-bot-token>
    TELEGRAM_CHAT_ID=<your-chat-id>

  Quick setup:
    1. Open Telegram → search @BotFather → send /newbot
    2. Follow prompts, copy the bot token
    3. Send any message to your new bot
    4. Visit https://api.telegram.org/bot<TOKEN>/getUpdates
    5. Find "chat":{"id": <number>} — that's your chat_id
"""


def send_telegram(msg: str) -> None:
    if not TG_TOKEN or not TG_CHAT:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": msg, "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception as exc:
        print(f"{DIM}  [tg] send failed: {exc}{RST}")


# ─── Signal Log ──────────────────────────────────────────────────────────


def log_signal(
    sym, sec, direction, price, sl, tp, atr, qty, rsi, vwap,
    vol_ratio, htf_val, regime_val, ts=None, phase="",
) -> None:
    """Append one signal row to the daily CSV log file."""
    LOG_DIR.mkdir(exist_ok=True)
    stamp = ts or datetime.now()
    log_date = stamp.strftime("%Y-%m-%d")
    log_file = LOG_DIR / f"signals_{log_date}.csv"

    is_new = not log_file.exists()
    trend = ""
    if regime_val is not None:
        trend = "Bullish" if price > regime_val else "Bearish"

    row = {
        "date": log_date,
        "time": stamp.strftime("%H:%M:%S"),
        "symbol": sym,
        "sector": sec,
        "direction": direction,
        "price": round(price, 2),
        "sl": round(sl, 2),
        "tp": round(tp, 2),
        "atr": round(atr, 2),
        "qty": qty,
        "risk": round(qty * atr, 2),
        "vwap": round(vwap, 2),
        "rsi": round(rsi, 1),
        "vol_ratio": round(vol_ratio, 1),
        "htf_ema": round(htf_val, 2) if htf_val is not None else "",
        "regime_ema": round(regime_val, 2) if regime_val is not None else "",
        "regime_trend": trend,
        "phase": phase,
    }

    with open(log_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_LOG_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


# ─── Indicator Engine ─────────────────────────────────────────────────────


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all strategy indicators on a 3-min OHLCV DataFrame.

    Replicates the exact indicator logic from strategy.py (backtrader)
    using pure pandas/numpy operations.
    """
    out = df.copy()

    tp = (out["high"] + out["low"] + out["close"]) / 3.0
    tp_vol = tp * out["volume"]
    day = out.index.date
    out["vwap"] = tp_vol.groupby(day).cumsum() / out["volume"].groupby(day).cumsum()

    out["ema9"] = out["close"].ewm(span=FAST_EMA, adjust=False).mean()
    out["ema21"] = out["close"].ewm(span=SLOW_EMA, adjust=False).mean()

    above = out["ema9"] > out["ema21"]
    prev_above = above.shift(1).fillna(False)
    out["cross_up"] = above & ~prev_above
    out["cross_dn"] = ~above & prev_above

    delta = out["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_g = gain.ewm(
        alpha=1.0 / RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False
    ).mean()
    avg_l = loss.ewm(
        alpha=1.0 / RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False
    ).mean()
    rs = avg_g / avg_l.replace(0, np.nan)
    out["rsi"] = 100.0 - 100.0 / (1.0 + rs)

    out["vol_sma"] = out["volume"].rolling(VOL_SMA).mean()

    pc = out["close"].shift(1)
    tr = np.maximum(out["high"], pc) - np.minimum(out["low"], pc)
    out["atr"] = tr.ewm(
        alpha=1.0 / ATR_PERIOD, min_periods=ATR_PERIOD, adjust=False
    ).mean()

    return out


def compute_htf_ema(df_3min: pd.DataFrame) -> pd.Series:
    """Resample 3-min to 15-min, compute 21-EMA on CLOSED candles only.

    Uses closed='right', label='right' so each 15-min bucket is labeled
    at its close time.  An incomplete bucket (e.g. only 2 of 5 bars at
    10:06) gets a label in the future (10:15), so reindex+ffill never
    picks it up — the 3-min rows keep using the last COMPLETED 15-min
    EMA value, eliminating intra-candle repainting / lookahead bias.
    """
    closed_15m = df_3min["close"].resample(
        "15min", closed="right", label="right"
    ).last().dropna()
    ema = closed_15m.ewm(span=HTF_EMA_PERIOD, adjust=False).mean()
    return ema.reindex(df_3min.index, method="ffill")


def compute_regime_ema(df_intraday: pd.DataFrame) -> float | None:
    """Derive Daily 21-EMA from intraday data (last close per trading day)."""
    daily_closes = df_intraday.groupby(df_intraday.index.date)["close"].last()
    if len(daily_closes) < REGIME_EMA_PERIOD:
        return None
    ema = daily_closes.ewm(span=REGIME_EMA_PERIOD, adjust=False).mean()
    return float(ema.iloc[-1])


# ─── Signal Detection ─────────────────────────────────────────────────────


def check_signal(row, htf_val, use_htf: bool, regime_val=None, vwap_ext_atr: float = 0) -> str | None:
    """Evaluate entry conditions on a single bar. Returns 'LONG', 'SHORT', or None.

    vwap_ext_atr: if > 0, reject entries where |close - VWAP| > vwap_ext_atr * ATR.
    """
    c, vwap = row["close"], row["vwap"]
    rsi, vol, vsma, atr = row["rsi"], row["volume"], row["vol_sma"], row["atr"]

    if pd.isna(rsi) or pd.isna(vsma) or pd.isna(atr) or atr <= 0 or pd.isna(vwap):
        return None

    allow_long = allow_short = True

    if regime_val is not None:
        allow_long = c > regime_val
        allow_short = c < regime_val

    if use_htf and htf_val is not None and not pd.isna(htf_val):
        allow_long = allow_long and c > htf_val
        allow_short = allow_short and c < htf_val

    if vwap_ext_atr > 0:
        if (c - vwap) > vwap_ext_atr * atr:
            allow_long = False
        if (vwap - c) > vwap_ext_atr * atr:
            allow_short = False

    if (
        allow_long
        and c > vwap
        and row["cross_up"]
        and vol > vsma
        and LONG_RSI_RANGE[0] <= rsi <= LONG_RSI_RANGE[1]
    ):
        return "LONG"

    if (
        allow_short
        and c < vwap
        and row["cross_dn"]
        and vol > vsma
        and SHORT_RSI_RANGE[0] <= rsi <= SHORT_RSI_RANGE[1]
    ):
        return "SHORT"

    return None


def calc_qty(capital: float, atr: float, price: float) -> int:
    risk_q = int(capital * RISK_PCT / atr)
    max_q = int(capital * LEV_CAP / price)
    return max(min(risk_q, max_q), 1)


# ─── Display ──────────────────────────────────────────────────────────────


def print_signal(
    sym, sec, direction, price, atr, qty, sl, tp,
    rsi, vwap, vol_ratio, htf_val, use_htf, regime_val=None, ts=None,
):
    clr = GRN if direction == "LONG" else RED
    arrow = "▲" if direction == "LONG" else "▼"
    stamp = (ts or datetime.now()).strftime("%H:%M:%S")
    print(f"\n{'=' * 60}")
    print(f"{clr}{BLD}  {arrow} {direction} — {sym} ({sec})  [{stamp}]{RST}")
    print(f"{'=' * 60}")
    print(f"  Price  : ₹{price:,.2f}")
    print(f"  SL     : ₹{sl:,.2f}  (1× ATR = ₹{atr:.2f})")
    print(f"  TP     : ₹{tp:,.2f}  (RR 1:{RR})")
    print(f"  Qty    : {qty}  (risk ₹{qty * atr:,.0f})")
    print(f"  {'─' * 25}")
    print(f"  VWAP   : ₹{vwap:,.2f}")
    print(f"  RSI(9) : {rsi:.1f}")
    print(f"  Vol    : {vol_ratio:.1f}× avg")
    if use_htf and htf_val is not None:
        print(f"  HTF EMA: ₹{htf_val:,.2f}")
    if regime_val is not None:
        trend = "Bullish" if price > regime_val else "Bearish"
        print(f"  Regime : ₹{regime_val:,.2f} ({trend})")
    print(f"{'=' * 60}\n")


def format_tg_signal(sym, sec, direction, price, atr, qty, sl, tp, rsi, vwap, vol_ratio):
    icon = "\U0001f7e2" if direction == "LONG" else "\U0001f534"
    return (
        f"{icon} *{direction} — {sym}* ({sec})\n\n"
        f"Price: ₹{price:,.2f}\n"
        f"SL: ₹{sl:,.2f} | TP: ₹{tp:,.2f}\n"
        f"Qty: {qty} | Risk: ₹{qty * atr:,.0f}\n"
        f"RR: 1:{RR}\n\n"
        f"VWAP: ₹{vwap:,.2f} | RSI: {rsi:.1f}\n"
        f"Vol: {vol_ratio:.1f}× avg"
    )


# ─── Stagnation Exit ──────────────────────────────────────────────────────


def _check_bracket_hit(pos: dict, bar_high: float, bar_low: float) -> str | None:
    """Return 'SL' or 'TP' if the bar's range would have triggered either."""
    if pos["direction"] == "LONG":
        if bar_low <= pos["sl"]:
            return "SL"
        if bar_high >= pos["tp"]:
            return "TP"
    else:
        if bar_high >= pos["sl"]:
            return "SL"
        if bar_low <= pos["tp"]:
            return "TP"
    return None


def _is_stagnant(
    sym: str, direction: str, entry_price: float,
    entry_atr: float, current_close: float, bars_held: int,
) -> bool:
    if sym in STAGNATION_SKIP:
        return False
    if bars_held < STAG_MAX_BARS:
        return False
    unrealized = (current_close - entry_price) if direction == "LONG" else (entry_price - current_close)
    return unrealized < STAG_MIN_PROFIT_ATR * entry_atr


def print_exit_signal(sym, sec, direction, entry_price, current_price, atr, unrealized, ts=None):
    clr = YLW
    stamp = (ts or datetime.now()).strftime("%H:%M:%S")
    pnl_sign = "+" if unrealized >= 0 else ""
    print(f"\n{'=' * 60}")
    print(f"{clr}{BLD}  ⚠ EXIT — {sym} ({sec})  [{stamp}]{RST}")
    print(f"{'=' * 60}")
    print(f"  Reason : Stagnation (< {STAG_MIN_PROFIT_ATR}× ATR after {STAG_MAX_BARS} bars)")
    print(f"  Side   : Close {direction} position")
    print(f"  Entry  : ₹{entry_price:,.2f}")
    print(f"  Now    : ₹{current_price:,.2f}")
    print(f"  P&L    : {pnl_sign}₹{unrealized:,.2f}  ({pnl_sign}{unrealized / atr:.2f}× ATR)")
    print(f"  Action : Cancel bracket order → close at market")
    print(f"{'=' * 60}\n")


def format_tg_exit(sym, sec, direction, entry_price, current_price, unrealized):
    pnl_sign = "+" if unrealized >= 0 else ""
    return (
        f"\u26a0\ufe0f *EXIT — {sym}* ({sec})\n\n"
        f"Reason: Stagnation ({STAG_MAX_BARS} bars, < {STAG_MIN_PROFIT_ATR}× ATR)\n"
        f"Close {direction} position\n"
        f"Entry: ₹{entry_price:,.2f} → Now: ₹{current_price:,.2f}\n"
        f"P&L: {pnl_sign}₹{unrealized:,.2f}\n\n"
        f"Cancel bracket → close at market"
    )


def print_dashboard(rows: list[dict], phase: str) -> None:
    now = datetime.now()
    phase_lbl = PHASES.get(phase, phase)
    if phase in ("ACTIVE_AM", "ACTIVE_PM"):
        phase_str = f"{GRN}{phase_lbl}{RST}"
    elif phase in ("SILENT", "MIDDAY"):
        phase_str = f"{YLW}{phase_lbl}{RST}"
    else:
        phase_str = f"{DIM}{phase_lbl}{RST}"

    print(f"\n{CYN}[{now:%H:%M:%S}]{RST} {phase_str}")
    print(
        f"  {'Symbol':<11} {'Close':>9}  {'VWAP':>9}  {'EMA':>4}  "
        f"{'RSI':>5}  {'Vol':>5}  {'HTF':>3}  {'Rgm':>3}  Signal"
    )
    print(f"  {'─' * 73}")
    for r in rows:
        cross = "▲" if r.get("cross_up") else ("▼" if r.get("cross_dn") else "·")
        rsi_s = f"{r['rsi']:.0f}" if not pd.isna(r.get("rsi", float("nan"))) else "  -"
        vr = r.get("vol_r", float("nan"))
        vol_s = f"{vr:.1f}x" if not pd.isna(vr) else "  -"

        htf_s = ""
        if r["use_htf"]:
            hv = r.get("htf_val")
            htf_s = "↑" if hv and r["close"] > hv else "↓"

        rgm = ""
        rv = r.get("regime_val")
        if rv is not None:
            rgm = "↑" if r["close"] > rv else "↓"

        sig = ""
        if r.get("signal"):
            c = GRN if r["signal"] == "LONG" else RED
            sig = f"{c}{BLD}{r['signal']}{RST}"

        print(
            f"  {r['sym']:<11} ₹{r['close']:>8,.2f}  ₹{r['vwap']:>8,.2f}  "
            f"{cross:>4}  {rsi_s:>5}  {vol_s:>5}  {htf_s:>3}  {rgm:>3}  {sig}"
        )


# ─── Timing Helpers ────────────────────────────────────────────────────────


def in_entry_window(t: time) -> bool:
    return (ENTRY_AM[0] <= t <= ENTRY_AM[1]) or (ENTRY_PM[0] <= t <= ENTRY_PM[1])


def next_candle_time() -> datetime:
    """Return next 3-min candle boundary + 2 seconds (HH:MM:02).

    Candles are on the 09:15 grid: close at 09:18, 09:21, ...
    We fetch at 09:18:02, 09:21:02, etc.
    """
    now = datetime.now()
    base = now.replace(hour=9, minute=15, second=0, microsecond=0)
    elapsed = (now - base).total_seconds()
    if elapsed < 0:
        return base + timedelta(seconds=180 + POLL_BUFFER_SEC)
    completed = int(elapsed / 180)
    next_close = base + timedelta(seconds=(completed + 1) * 180)
    return next_close + timedelta(seconds=POLL_BUFFER_SEC)


# ─── Data Loading ─────────────────────────────────────────────────────────

_FS_SAFE = str.maketrans({"&": ""})


def _cache_path(symbol: str) -> Path:
    return DATA_DIR / f"{symbol.translate(_FS_SAFE)}_3min.csv"


def fetch_regime_emas(client: AngelOneClient) -> dict[str, float]:
    """Fetch daily candles for each stock and compute Daily 21-EMA (regime gate)."""
    regime: dict[str, float] = {}
    end = datetime.now()
    start = end - timedelta(days=REGIME_FETCH_DAYS)

    for sym, _ in STOCKS:
        try:
            df = client.fetch_candles(
                sym,
                start.strftime("%Y-%m-%d 09:15"),
                end.strftime("%Y-%m-%d 15:30"),
                interval="ONE_DAY",
            )
            if len(df) >= REGIME_EMA_PERIOD:
                ema = df["close"].ewm(span=REGIME_EMA_PERIOD, adjust=False).mean()
                regime[sym] = float(ema.iloc[-1])
                last_close = float(df["close"].iloc[-1])
                direction = "Bullish" if last_close > regime[sym] else "Bearish"
                trend = f"{GRN}↑ {direction}{RST}" if direction == "Bullish" else f"{RED}↓ {direction}{RST}"
                print(f"  {sym:<11} Daily EMA(21) ₹{regime[sym]:>9,.2f}  {trend}")
                flog.info(
                    "REGIME  %-11s  EMA=%.2f  Close=%.2f  %s",
                    sym, regime[sym], last_close, direction,
                )
            else:
                print(f"  {YLW}{sym:<11} only {len(df)} daily bars (need {REGIME_EMA_PERIOD}){RST}")
                flog.warning("REGIME  %-11s  only %d daily bars (need %d)", sym, len(df), REGIME_EMA_PERIOD)
            _time.sleep(0.3)
        except Exception as exc:
            print(f"  {RED}{sym:<11} daily fetch failed: {exc}{RST}")
            flog.error("REGIME  %-11s  fetch failed: %s", sym, exc)
    return regime


def load_warmup(client: AngelOneClient | None) -> dict[str, pd.DataFrame]:
    """Load 3-min warmup data: prefer fresh API data, fall back to cache."""
    data: dict[str, pd.DataFrame] = {}
    today = datetime.now().date()
    for sym, _ in STOCKS:
        cache = _cache_path(sym)

        if cache.exists():
            try:
                df = load_csv(str(cache))
                if df.index[-1].date() >= today or client is None:
                    data[sym] = df
                    print(f"  {sym:<11} {len(df):>6} bars  (cache: {df.index[-1]:%Y-%m-%d})")
                    flog.info("WARMUP  %-11s  %d bars  cache=%s", sym, len(df), df.index[-1].date())
                    continue
            except Exception:
                pass

        if client is not None:
            try:
                df = client.fetch_history(sym, days=WARMUP_3MIN_DAYS)
                data[sym] = df
                print(
                    f"  {sym:<11} {len(df):>6} bars  "
                    f"(API: {df.index[0]:%m-%d} → {df.index[-1]:%m-%d %H:%M})"
                )
                flog.info(
                    "WARMUP  %-11s  %d bars  API %s→%s",
                    sym, len(df), df.index[0].date(), df.index[-1],
                )
            except Exception as exc:
                if cache.exists():
                    try:
                        df = load_csv(str(cache))
                        data[sym] = df
                        print(f"  {YLW}{sym:<11} {len(df):>6} bars  (stale cache: {df.index[-1]:%Y-%m-%d}){RST}")
                        flog.warning("WARMUP  %-11s  %d bars  stale_cache=%s (API: %s)", sym, len(df), df.index[-1].date(), exc)
                        continue
                    except Exception:
                        pass
                print(f"  {RED}{sym:<11} FAILED: {exc}{RST}")
                flog.error("WARMUP  %-11s  FAILED: %s", sym, exc)
            _time.sleep(0.3)
        else:
            print(f"  {YLW}{sym:<11} no data{RST}")
            flog.warning("WARMUP  %-11s  no data (no API, no cache)", sym)
    return data


# ─── Signal Scan (shared by dry-run and live) ─────────────────────────────


def scan_bar(sym, sec, df, use_htf, htf_series, bar_idx, capital, regime_val=None):
    """Check one bar for a signal. Returns a result dict."""
    row = df.iloc[bar_idx]
    ts = df.index[bar_idx]
    htf_val = None
    if htf_series is not None and ts in htf_series.index:
        htf_val = htf_series.loc[ts]

    ext = VWAP_EXT_ATR if sym in VWAP_EXT_STOCKS else 0
    sig = check_signal(row, htf_val, use_htf, regime_val=regime_val, vwap_ext_atr=ext)
    result = {
        "sym": sym,
        "sec": sec,
        "close": row["close"],
        "vwap": row["vwap"],
        "cross_up": bool(row.get("cross_up", False)),
        "cross_dn": bool(row.get("cross_dn", False)),
        "rsi": row["rsi"],
        "vol_r": (
            row["volume"] / row["vol_sma"]
            if row["vol_sma"] > 0 and not pd.isna(row["vol_sma"])
            else float("nan")
        ),
        "use_htf": use_htf,
        "htf_val": htf_val,
        "regime_val": regime_val,
        "signal": sig,
    }

    if sig:
        p, a = row["close"], row["atr"]
        q = calc_qty(capital, a, p)
        sl = p - a if sig == "LONG" else p + a
        tp = p + a * RR if sig == "LONG" else p - a * RR
        vr = result["vol_r"] if not pd.isna(result["vol_r"]) else 0
        result.update(price=p, atr=a, qty=q, sl=sl, tp=tp, vol_ratio=vr, ts=ts)

    return result


# ─── Fetch helper for live loop ───────────────────────────────────────────


def _fetch_latest(client, sym, stock_data, now):
    """Fetch the latest few candles and merge into existing data.

    Drops the current incomplete candle so indicators only use closed bars.
    """
    window_start = (now - timedelta(minutes=6)).strftime("%Y-%m-%d %H:%M")
    window_end = now.strftime("%Y-%m-%d %H:%M")
    fresh = client.fetch_candles(sym, window_start, window_end)

    base = now.replace(hour=9, minute=15, second=0, microsecond=0)
    elapsed = (now - base).total_seconds()
    if elapsed >= 0:
        current_candle_open = base + timedelta(seconds=int(elapsed / 180) * 180)
        fresh = fresh[fresh.index < current_candle_open]

    if fresh.empty:
        return

    combined = pd.concat([stock_data[sym], fresh])
    stock_data[sym] = combined[~combined.index.duplicated(keep="last")].sort_index()


# ─── Dry Run ──────────────────────────────────────────────────────────────


def dry_run(capital: float) -> None:
    print(f"\n{CYN}[data]{RST} Loading data for {len(STOCKS)} stocks...")
    client = None
    try:
        client = AngelOneClient()
        client.connect()
    except Exception as exc:
        print(f"{YLW}[api] Could not connect ({exc}) — using cache only{RST}")

    data = load_warmup(client)
    if not data:
        print(f"{RED}No data available. Run a backtest first to populate the cache.{RST}")
        return

    print(f"\n{CYN}[regime]{RST} Computing Daily 21-EMA regime filter...")
    regime_emas: dict[str, float] = {}
    if client:
        regime_emas = fetch_regime_emas(client)
    else:
        for sym, _ in STOCKS:
            if sym in data:
                val = compute_regime_ema(data[sym])
                if val is not None:
                    regime_emas[sym] = val
                    print(f"  {sym:<11} Daily EMA(21) ₹{val:>9,.2f}  (from cache)")

    missing_regime = [s for s, _ in STOCKS if s in data and s not in regime_emas]
    if missing_regime:
        print(
            f"\n{YLW}  ⚠ Regime filter OFF for {len(missing_regime)} stock(s) "
            f"(need {REGIME_EMA_PERIOD} daily bars, cache has < {REGIME_EMA_PERIOD}):"
            f"\n    {', '.join(missing_regime)}{RST}"
        )

    ref_sym = next(iter(data))
    scan_date = data[ref_sym].index[-1].date()
    print(f"\n{YLW}{BLD}── DRY RUN: Scanning {scan_date} for signals ──{RST}\n")

    signals_found = 0
    stag_exits = 0
    cooldowns: dict[str, datetime] = {}
    for sym, sec in STOCKS:
        if sym not in data:
            continue
        df = compute_indicators(data[sym])
        use_htf = sym in HTF_STOCKS
        htf_s = compute_htf_ema(data[sym]) if use_htf else None
        if use_htf and htf_s is not None:
            n_15m = htf_s.dropna().shape[0]
            if n_15m < HTF_MIN_BARS_15M:
                flog.warning(
                    "HTF disabled %-11s  only %d/%d 15-min bars (dry-run)",
                    sym, n_15m, HTF_MIN_BARS_15M,
                )
                use_htf = False
                htf_s = None
        regime_val = regime_emas.get(sym)

        current_pos: dict | None = None
        bars_since_entry = 0

        day_mask = df.index.date == scan_date
        for i in df.index[day_mask]:
            idx = df.index.get_loc(i)
            row = df.iloc[idx]

            if current_pos is not None:
                bars_since_entry += 1
                hit = _check_bracket_hit(current_pos, row["high"], row["low"])
                if hit:
                    current_pos = None
                    bars_since_entry = 0
                elif _is_stagnant(sym, current_pos["direction"], current_pos["entry_price"],
                                  current_pos["entry_atr"], row["close"], bars_since_entry):
                    c = row["close"]
                    unrealized = (
                        (c - current_pos["entry_price"]) if current_pos["direction"] == "LONG"
                        else (current_pos["entry_price"] - c)
                    )
                    stag_exits += 1
                    print_exit_signal(
                        sym, sec, current_pos["direction"],
                        current_pos["entry_price"], c, current_pos["entry_atr"],
                        unrealized, ts=i,
                    )
                    current_pos = None
                    bars_since_entry = 0
                continue

            if not in_entry_window(i.time()):
                continue
            on_cooldown = sym in cooldowns and i < cooldowns[sym]
            if on_cooldown:
                continue

            result = scan_bar(sym, sec, df, use_htf, htf_s, idx, capital, regime_val)
            if result["signal"]:
                signals_found += 1
                cooldowns[sym] = i + COOLDOWN
                current_pos = {
                    "direction": result["signal"],
                    "entry_price": result["price"],
                    "entry_atr": result["atr"],
                    "sl": result["sl"],
                    "tp": result["tp"],
                }
                bars_since_entry = 0
                r = result
                print_signal(
                    sym, sec, r["signal"], r["price"], r["atr"], r["qty"],
                    r["sl"], r["tp"], r["rsi"], r["vwap"], r["vol_ratio"],
                    r["htf_val"], use_htf, regime_val=regime_val, ts=r["ts"],
                )
                phase = get_phase(i.time())
                log_signal(
                    sym, sec, r["signal"], r["price"], r["sl"], r["tp"],
                    r["atr"], r["qty"], r["rsi"], r["vwap"], r["vol_ratio"],
                    r["htf_val"], regime_val, ts=i, phase=phase,
                )

    log_file = LOG_DIR / f"signals_{scan_date}.csv"
    print(f"\n{CYN}── Dry run complete: {signals_found} signal(s), {stag_exits} stagnation exit(s) on {scan_date} ──{RST}")
    if signals_found:
        print(f"{DIM}  Log saved: {log_file}{RST}")
    if stag_exits:
        skip_note = f"  (HDFCLIFE exempted)" if "HDFCLIFE" in STAGNATION_SKIP else ""
        print(f"{DIM}  Stagnation exit: {STAG_MAX_BARS} bars, < {STAG_MIN_PROFIT_ATR}× ATR{skip_note}{RST}")


# ─── Live Loop (Phase-Based Architecture) ─────────────────────────────────


def live(capital: float) -> None:
    flog.info("=" * 70)
    flog.info("SESSION START  Capital=%.0f  RR=1:%.1f  Risk=%.0f%%", capital, RR, RISK_PCT * 100)
    flog.info("Stocks: %s", ", ".join(s for s, _ in STOCKS))
    flog.info("HTF stocks: %s", ", ".join(sorted(HTF_STOCKS)))

    if TG_TOKEN and TG_CHAT:
        print(f"{GRN}[tg] Telegram notifications enabled{RST}")
        flog.info("Telegram: enabled")
    else:
        print(f"{YLW}[tg] Not configured — console only{RST}")
        print(_TELEGRAM_SETUP)
        flog.info("Telegram: disabled")

    # ── Phase 1: Warm-Up ──────────────────────────────────────────────
    print(f"\n{CYN}{BLD}Phase 1 · Warm-Up{RST}")
    print(f"{CYN}[api] Connecting to Angel One...{RST}")
    flog.info("PHASE 1  Warm-Up — connecting to Angel One")
    client = AngelOneClient()
    client.connect()
    flog.info("API connected")

    print(f"\n{CYN}[regime] Fetching daily candles → Daily 21-EMA regime filter{RST}")
    flog.info("Fetching regime EMAs (%d-day daily candles)", REGIME_FETCH_DAYS)
    regime_emas = fetch_regime_emas(client)

    print(f"\n{CYN}[data] Fetching {WARMUP_3MIN_DAYS}-day 3-min candles (indicator warmup){RST}")
    flog.info("Loading %d-day 3-min warmup data", WARMUP_3MIN_DAYS)
    stock_data = load_warmup(client)
    loaded = len(stock_data)

    if loaded == 0:
        print(f"{RED}No data loaded. Exiting.{RST}")
        flog.error("No data loaded — exiting")
        return

    syms = ", ".join(s for s, _ in STOCKS if s in stock_data)
    regime_count = len(regime_emas)
    htf_count = sum(1 for s, _ in STOCKS if s in HTF_STOCKS and s in stock_data)

    flog.info(
        "Warm-up complete: %d stocks loaded, %d HTF, %d regime",
        loaded, htf_count, regime_count,
    )

    HTF_BACKFILL_DAYS = 10
    for sym, _ in STOCKS:
        if sym in HTF_STOCKS and sym in stock_data:
            _htf_check = compute_htf_ema(stock_data[sym])
            _n = _htf_check.dropna().shape[0] if _htf_check is not None else 0
            if _n < HTF_MIN_BARS_15M and client is not None:
                print(f"  HTF check  {sym:<11}  {_n:>3} / {HTF_MIN_BARS_15M} 15-min bars  [{YLW}LOW — backfilling {HTF_BACKFILL_DAYS}d...{RST}]")
                flog.warning("HTF CHECK  %-11s  %d/%d 15-min bars  LOW — backfilling %dd", sym, _n, HTF_MIN_BARS_15M, HTF_BACKFILL_DAYS)
                try:
                    df_deep = client.fetch_history(sym, days=HTF_BACKFILL_DAYS)
                    combined = pd.concat([df_deep, stock_data[sym]])
                    stock_data[sym] = combined[~combined.index.duplicated(keep="last")].sort_index()
                    _htf_check = compute_htf_ema(stock_data[sym])
                    _n = _htf_check.dropna().shape[0] if _htf_check is not None else 0
                    flog.info("HTF BACKFILL  %-11s  %d bars total  %d 15-min bars", sym, len(stock_data[sym]), _n)
                except Exception as exc:
                    flog.error("HTF BACKFILL  %-11s  FAILED: %s", sym, exc)
                _time.sleep(0.3)
            ok = _n >= HTF_MIN_BARS_15M
            tag = f"{GRN}OK{RST}" if ok else f"{YLW}LOW — HTF filter will be disabled{RST}"
            print(f"  HTF check  {sym:<11}  {_n:>3} / {HTF_MIN_BARS_15M} 15-min bars  [{tag}]")
            flog.info("HTF CHECK  %-11s  %d/%d 15-min bars  %s", sym, _n, HTF_MIN_BARS_15M, "OK" if ok else "LOW")

    print(f"\n{BLD}{'═' * 60}")
    print(f"  LIVE SIGNAL MONITOR")
    print(f"  Capital : ₹{capital:,.0f}  |  RR 1:{RR}  |  Risk {RISK_PCT*100:.0f}%")
    print(f"  Stocks  : {loaded} loaded  |  {htf_count} HTF  |  {regime_count} regime")
    print(f"  {'─' * 56}")
    print(f"  09:14  Phase 1  Warm-Up       ✓ done")
    print(f"  09:15  Phase 2  Silent         track only, no signals")
    print(f"  10:00  Phase 3  Morning        signals ON")
    print(f"  12:00  Phase 4  Midday         track only, no signals")
    print(f"  14:00  Phase 5  Afternoon      signals ON")
    print(f"  15:00  Phase 6  Shutdown       stop")
    print(f"{'═' * 60}{RST}\n")

    send_telegram(
        f"\U0001f514 *Signal Monitor Started*\n"
        f"Capital: ₹{capital:,.0f}\n"
        f"Stocks: {syms}\n"
        f"Regime: Daily 21-EMA ({regime_count} stocks)\n"
        f"Schedule: 10:00–12:00, 14:00–15:00"
    )

    cooldowns: dict[str, datetime] = {}
    open_positions: dict[str, dict] = {}
    signal_counts: dict[tuple[str, str], int] = {}
    today_signals: list[dict] = []
    today_exits: list[dict] = []
    current_date = datetime.now().date()
    last_phase = ""
    scan_count = 0
    last_scan_candle = -1

    try:
        while True:
            now = datetime.now()
            t = now.time()
            phase = get_phase(t)

            # Day rollover
            if now.date() != current_date:
                flog.info("DAY ROLLOVER  %s → %s", current_date, now.date())
                current_date = now.date()
                cooldowns.clear()
                open_positions.clear()
                signal_counts.clear()
                today_signals.clear()
                today_exits.clear()
                scan_count = 0
                last_scan_candle = -1
                last_phase = ""
                print(f"\n{CYN}[new day] {current_date}{RST}")
                print(f"{CYN}[regime] Refreshing daily EMAs...{RST}")
                try:
                    regime_emas = fetch_regime_emas(client)
                except Exception as exc:
                    print(f"{YLW}  regime refresh failed — using yesterday's values{RST}")
                    flog.error("Regime refresh failed: %s — using previous values", exc)

            # Phase transition announcement
            if phase != last_phase:
                prev_phase = last_phase
                last_phase = phase
                lbl = PHASES.get(phase, phase)
                flog.info("PHASE  %s → %s", prev_phase or "START", phase)
                if phase in ("ACTIVE_AM", "ACTIVE_PM"):
                    print(f"\n{GRN}{BLD}▶ {lbl} — signals ENABLED{RST}")
                    send_telegram(f"\u25b6\ufe0f *{lbl}* — signals enabled")
                elif phase == "SHUTDOWN":
                    n = len(today_signals)
                    ne = len(today_exits)
                    log_file = LOG_DIR / f"signals_{current_date}.csv"
                    print(f"\n{DIM}[{now:%H:%M}] {lbl}. {n} signal(s), {ne} stagnation exit(s) today.{RST}")
                    flog.info("=" * 70)
                    flog.info("SHUTDOWN  %d signal(s), %d stagnation exit(s), %d scan cycles", n, ne, scan_count)
                    if today_signals:
                        for s in today_signals:
                            print(f"  {s['time']:%H:%M}  {s['dir']:>5}  {s['sym']:<11}  @ ₹{s['price']:,.2f}")
                            flog.info(
                                "  SUMMARY  %s  %5s  %-11s  @ %.2f  SL=%.2f  TP=%.2f",
                                s["time"].strftime("%H:%M"), s["dir"], s["sym"],
                                s["price"], s["sl"], s["tp"],
                            )
                        print(f"{DIM}  Log saved: {log_file}{RST}")
                    if today_exits:
                        print(f"  {YLW}Stagnation exits:{RST}")
                        for e in today_exits:
                            pnl_sign = "+" if e["unrealized"] >= 0 else ""
                            print(f"  {e['time']:%H:%M}  EXIT  {e['sym']:<11}  {pnl_sign}₹{e['unrealized']:,.2f}")
                            flog.info(
                                "  STAG_EXIT  %s  %-11s  entry=%.2f  exit=%.2f  pnl=%.2f",
                                e["time"].strftime("%H:%M"), e["sym"],
                                e["entry"], e["exit"], e["unrealized"],
                            )
                    flog.info("=" * 70)
                    send_telegram(f"\U0001f4f4 Shutdown. {n} signal(s), {ne} stagnation exit(s) today.")
                    break
                elif phase == "SILENT":
                    print(f"\n{YLW}▶ {lbl} — tracking indicators, no signals{RST}")
                elif phase == "MIDDAY":
                    print(f"\n{YLW}▶ {lbl} — indicators updating, signals blocked{RST}")

            # Pre-market wait
            if phase == "PRE_MARKET":
                wait = int((datetime.combine(now.date(), MKT_OPEN) - now).total_seconds())
                print(f"{DIM}[{now:%H:%M}] Pre-market. Opens in {wait // 60}m...{RST}")
                _time.sleep(min(wait, 60))
                continue

            # Align to 3-min candle grid (09:15, 09:18, 09:21, ...)
            now_ts = datetime.now()
            _base = now_ts.replace(hour=9, minute=15, second=0, microsecond=0)
            _elapsed = (now_ts - _base).total_seconds()
            _candle_idx = int(_elapsed / 180) if _elapsed >= 0 else -1

            if _candle_idx <= last_scan_candle:
                nxt = _base + timedelta(
                    seconds=(last_scan_candle + 1) * 180 + POLL_BUFFER_SEC
                )
                wait_s = max(0, (nxt - datetime.now()).total_seconds())
                lbl = PHASES.get(phase, "")
                m, s = divmod(int(wait_s), 60)
                sys.stdout.write(
                    f"\r{DIM}[{now:%H:%M:%S}] {lbl} — "
                    f"next scan {nxt:%H:%M:%S} ({m}m {s}s)   {RST}"
                )
                sys.stdout.flush()
                _time.sleep(min(max(wait_s, 0.5), 30))
                continue

            _poll_at = _base + timedelta(
                seconds=_candle_idx * 180 + POLL_BUFFER_SEC
            )
            _wait_poll = (_poll_at - datetime.now()).total_seconds()
            if _wait_poll > 0:
                _time.sleep(_wait_poll)

            last_scan_candle = _candle_idx

            # ── Scan cycle ────────────────────────────────────────────
            scan_count += 1
            is_active = phase in ("ACTIVE_AM", "ACTIVE_PM")
            dashboard: list[dict] = []
            flog.info(
                "SCAN #%d  %s  phase=%s  active=%s",
                scan_count, now.strftime("%H:%M:%S"), phase, is_active,
            )

            for sym, sec in STOCKS:
                if sym not in stock_data:
                    continue

                try:
                    _fetch_latest(client, sym, stock_data, now)
                except Exception as exc:
                    if "invalid" in str(exc).lower() or "session" in str(exc).lower():
                        flog.warning("API  %-11s  session expired, reconnecting", sym)
                        try:
                            client.connect()
                            _fetch_latest(client, sym, stock_data, now)
                            flog.info("API  %-11s  reconnected OK", sym)
                        except Exception as exc2:
                            print(f"  {RED}{sym}: reconnect failed{RST}")
                            flog.error("API  %-11s  reconnect FAILED: %s", sym, exc2)
                            continue
                    else:
                        print(f"  {DIM}{sym}: {exc}{RST}")
                        flog.warning("API  %-11s  fetch error: %s", sym, exc)
                        continue

                _time.sleep(0.35)

                df = compute_indicators(stock_data[sym])
                use_htf = sym in HTF_STOCKS
                htf_s = compute_htf_ema(stock_data[sym]) if use_htf else None
                if use_htf and htf_s is not None:
                    n_15m = htf_s.dropna().shape[0]
                    if n_15m < HTF_MIN_BARS_15M:
                        flog.warning(
                            "HTF disabled %-11s  only %d/%d 15-min bars",
                            sym, n_15m, HTF_MIN_BARS_15M,
                        )
                        use_htf = False
                        htf_s = None
                regime_val = regime_emas.get(sym)
                last_row = df.iloc[-1]

                if sym in open_positions:
                    pos = open_positions[sym]
                    hit = _check_bracket_hit(pos, last_row["high"], last_row["low"])
                    if hit:
                        flog.info("BRACKET  %-11s  %s hit — position closed", sym, hit)
                        del open_positions[sym]
                    else:
                        bars_held = scan_count - pos["entry_scan"]
                        if _is_stagnant(sym, pos["direction"], pos["entry_price"],
                                        pos["entry_atr"], last_row["close"], bars_held):
                            c = last_row["close"]
                            unrealized = (
                                (c - pos["entry_price"]) if pos["direction"] == "LONG"
                                else (pos["entry_price"] - c)
                            )
                            print_exit_signal(
                                sym, sec, pos["direction"],
                                pos["entry_price"], c, pos["entry_atr"], unrealized,
                            )
                            send_telegram(format_tg_exit(
                                sym, sec, pos["direction"],
                                pos["entry_price"], c, unrealized,
                            ))
                            flog.info(
                                "STAGNATION EXIT  %-11s  %s  entry=%.2f  now=%.2f  unrealized=%.2f  bars=%d",
                                sym, pos["direction"], pos["entry_price"], c, unrealized, bars_held,
                            )
                            today_exits.append({
                                "sym": sym, "dir": pos["direction"],
                                "entry": pos["entry_price"], "exit": c,
                                "unrealized": unrealized, "time": now,
                            })
                            del open_positions[sym]

                result = scan_bar(
                    sym, sec, df, use_htf, htf_s, -1, capital, regime_val
                )

                on_cooldown = sym in cooldowns and now < cooldowns[sym]

                # Build log line for every stock in this scan
                r = result
                cross_str = "UP" if r.get("cross_up") else ("DN" if r.get("cross_dn") else "--")
                rsi_str = f"{r['rsi']:.0f}" if not pd.isna(r.get("rsi", float("nan"))) else "-"
                vr = r.get("vol_r", float("nan"))
                vol_str = f"{vr:.1f}x" if not pd.isna(vr) else "-"
                htf_str = f"{r['htf_val']:.2f}" if r.get("htf_val") is not None and not pd.isna(r.get("htf_val", float("nan"))) else "-"
                raw_sig = r.get("signal") or "-"

                already_in_position = sym in open_positions
                session_key = "AM" if phase == "ACTIVE_AM" else "PM"
                at_signal_cap = signal_counts.get((sym, session_key), 0) >= MAX_SIGNALS_PER_STOCK
                if is_active and result["signal"] and not on_cooldown and not already_in_position and not at_signal_cap:
                    print_signal(
                        sym, sec, r["signal"], r["price"], r["atr"], r["qty"],
                        r["sl"], r["tp"], r["rsi"], r["vwap"], r["vol_ratio"],
                        r["htf_val"], use_htf, regime_val=regime_val,
                    )
                    send_telegram(format_tg_signal(
                        sym, sec, r["signal"], r["price"], r["atr"], r["qty"],
                        r["sl"], r["tp"], r["rsi"], r["vwap"], r["vol_ratio"],
                    ))
                    log_signal(
                        sym, sec, r["signal"], r["price"], r["sl"], r["tp"],
                        r["atr"], r["qty"], r["rsi"], r["vwap"], r["vol_ratio"],
                        r["htf_val"], regime_val, phase=phase,
                    )
                    cooldowns[sym] = now + COOLDOWN
                    signal_counts[(sym, session_key)] = signal_counts.get((sym, session_key), 0) + 1
                    open_positions[sym] = {
                        "direction": r["signal"],
                        "entry_price": r["price"],
                        "entry_atr": r["atr"],
                        "sl": r["sl"],
                        "tp": r["tp"],
                        "entry_scan": scan_count,
                    }
                    today_signals.append({
                        "sym": sym, "dir": r["signal"],
                        "price": r["price"], "sl": r["sl"], "tp": r["tp"],
                        "time": now,
                    })
                    flog.info(
                        "  %-11s  C=%.2f  VWAP=%.2f  EMA=%s  RSI=%s  Vol=%s  HTF=%s  → ★ %s  SL=%.2f  TP=%.2f  Qty=%d",
                        sym, r["close"], r["vwap"], cross_str, rsi_str,
                        vol_str, htf_str, r["signal"], r["sl"], r["tp"], r["qty"],
                    )
                else:
                    reason = ""
                    if not is_active:
                        reason = "inactive_phase"
                    elif already_in_position:
                        reason = "in_position"
                    elif on_cooldown:
                        reason = f"cooldown_until_{cooldowns[sym]:%H:%M}"
                    elif at_signal_cap:
                        reason = f"max_signals_{session_key}({signal_counts.get((sym, session_key), 0)})"
                    elif raw_sig != "-":
                        reason = "blocked"
                    result["signal"] = None
                    flog.debug(
                        "  %-11s  C=%.2f  VWAP=%.2f  EMA=%s  RSI=%s  Vol=%s  HTF=%s  → %s  %s",
                        sym, r["close"], r["vwap"], cross_str, rsi_str,
                        vol_str, htf_str, raw_sig, reason,
                    )

                dashboard.append(result)

            print_dashboard(dashboard, phase)

    except KeyboardInterrupt:
        n = len(today_signals)
        flog.info("STOPPED MANUALLY  %d signal(s), %d scans", n, scan_count)
        for s in today_signals:
            flog.info(
                "  SUMMARY  %s  %5s  %-11s  @ %.2f",
                s["time"].strftime("%H:%M"), s["dir"], s["sym"], s["price"],
            )
        print(f"\n\n{YLW}[stopped] {n} signal(s) generated today:{RST}")
        for s in today_signals:
            print(f"  {s['time']:%H:%M}  {s['dir']:>5}  {s['sym']:<11}  @ ₹{s['price']:,.2f}")
        send_telegram(f"\u23f9 Monitor stopped manually. {n} signal(s) today.")


# ─── CLI Entry ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Live NSE Intraday Scalping Signal Generator",
    )
    parser.add_argument(
        "--capital", type=float, default=50_000,
        help="Trading capital in INR (default: 50000)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Scan last cached session for signals, then exit",
    )
    args = parser.parse_args()

    print(f"\n{BLD}NSE Intraday Scalping — Live Signal Generator{RST}")
    print(f"{DIM}EMA(9/21) + VWAP + RSI(9) + Vol + ATR | Regime: Daily 21-EMA")
    print(f"RR 1:{RR} | Risk {RISK_PCT * 100:.0f}% | Polls at HH:MM:02{RST}\n")

    if args.dry_run:
        dry_run(args.capital)
    else:
        live(args.capital)


if __name__ == "__main__":
    main()

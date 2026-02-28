"""
Live Signal Generator for NSE Intraday Scalping Strategy — HYBRID EDITION

Polls Angel One SmartAPI every 3 minutes (at HH:MM:01) for 10 stocks,
computes indicators in real-time (pandas/numpy), and sends BUY/SHORT
alerts to console + Telegram.

HYBRID STRATEGY FEATURES:
  1. 2-Bar Volume Confirmation — requires sustained volume over 2 bars
  2. Dynamic Body Ratio — 0.72 if ATR > 1.1x, else 0.70
  3. ATR-Based SL Adjustment — widens SL 1.2x in high vol, tightens 0.9x in low vol
  4. Morning-Only Trading — 10:00-12:00 (avoids afternoon chop)

Daily Lifecycle:
  Phase 1  09:14  Warm-Up   — connect, fetch 5 days of 3-min data
  Phase 2  09:15  Silent    — track indicators every 3 min, NO signals
  Phase 3  10:00  Active AM — morning trading window, signals enabled (HYBRID ON)
  Phase 4  12:00  Shutdown  — stop signals, track only until close

Entry A : PDL — Prev-Day Level Breakout (close crosses yesterday's H/L)
              + VWAP confirmation + RSI > 50 / < 50 + Volume > SMA
              SL = 1.5× ATR (dynamic) · max 1 per direction per day
Entry B : MOM — Momentum Breakout (2× sustained volume + 70%/72% body ratio)
              + VWAP confirmation + widened RSI (30-85 / 15-70)
              SL = 1.2× ATR (dynamic)
RR      : 1 : 1.75
Sizing  : 1% risk per trade, 5× leverage cap
Exit    : Stagnation exit if profit < 0.2×ATR after 8 bars
Guard   : Max 2 SL hits per stock per day — blocks further entries for that stock

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
    ("BAJAJ-AUTO", "Auto"),
    ("POWERGRID", "Power"),
    ("TATACONSUM", "Consumer"),
    ("INDIGO", "Aviation"),
    ("BAJFINANCE", "Finance"),
    ("HCLTECH", "IT"),
    ("JIOFIN", "Finance"),
    ("EICHERMOT", "Auto"),
    ("NESTLEIND", "Consumer"),
    ("BHARTIARTL", "Telecom"),
]

STAG_MAX_BARS = 8
STAG_MIN_PROFIT_ATR = 0.2


# ─── Strategy Parameters ──────────────────────────────────────────────────

RSI_PERIOD = 9
VOL_SMA = 20
ATR_PERIOD = 14

RR = 1.75
RISK_PCT = 0.01
LEV_CAP = 5.0

PDL_SL_MULT = 1.5
MOM_SL_MULT = 1.2
MOM_VOL_MULT = 2.0
MOM_BODY_RATIO = 0.70
MAX_SL_PER_DAY = 1
MOM_LONG_RSI = (30, 85)
MOM_SHORT_RSI = (15, 70)
PDL_LONG_RSI_MIN = 50
PDL_SHORT_RSI_MAX = 50


# ─── Time Rules & Phases ──────────────────────────────────────────────────

MKT_OPEN = time(9, 15)
ENTRY_AM = (time(10, 0), time(12, 0))
ENTRY_PM = None  # Morning-only trading (Hybrid Strategy)
MKT_CLOSE = time(15, 0)

COOLDOWN = timedelta(minutes=15)
MAX_SIGNALS_PER_STOCK = 2
WARMUP_3MIN_DAYS = 5
POLL_BUFFER_SEC = 1

DATA_DIR = Path(__file__).parent / "data"
LOG_DIR = Path(__file__).parent / "logs"
LIVE_LOG_DIR = LOG_DIR / "live_signals"
SIGNALS_DIR = LOG_DIR / "signals"

logging.getLogger().handlers = []
logging.getLogger().addHandler(logging.NullHandler())


def _init_file_logger() -> logging.Logger:
    LIVE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("live_signals")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if not logger.handlers:
        fh = logging.FileHandler(
            LIVE_LOG_DIR / f"live_{datetime.now():%Y-%m-%d}.log", encoding="utf-8",
        )
        fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(fh)
    return logger


flog = _init_file_logger()

_LOG_COLUMNS = [
    "date", "time", "symbol", "sector", "direction", "trigger", "price",
    "sl", "tp", "atr", "qty", "risk", "vwap", "rsi", "vol_ratio", "phase",
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
    sym, sec, direction, trigger, price, sl, tp, atr, qty, rsi, vwap,
    vol_ratio, ts=None, phase="",
) -> None:
    """Append one signal row to the daily CSV log file."""
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = ts or datetime.now()
    log_date = stamp.strftime("%Y-%m-%d")
    log_file = SIGNALS_DIR / f"signals_{log_date}.csv"

    is_new = not log_file.exists()
    row = {
        "date": log_date,
        "time": stamp.strftime("%H:%M:%S"),
        "symbol": sym,
        "sector": sec,
        "direction": direction,
        "trigger": trigger or "",
        "price": round(price, 2),
        "sl": round(sl, 2),
        "tp": round(tp, 2),
        "atr": round(atr, 2),
        "qty": qty,
        "risk": round(qty * atr, 2),
        "vwap": round(vwap, 2),
        "rsi": round(rsi, 1),
        "vol_ratio": round(vol_ratio, 1),
        "phase": phase,
    }

    with open(log_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_LOG_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


# ─── Indicator Engine ─────────────────────────────────────────────────────


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all strategy indicators on a 3-min OHLCV DataFrame."""
    out = df.copy()

    tp = (out["high"] + out["low"] + out["close"]) / 3.0
    tp_vol = tp * out["volume"]
    day = out.index.date
    out["vwap"] = tp_vol.groupby(day).cumsum() / out["volume"].groupby(day).cumsum()

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

    out["body"] = (out["close"] - out["open"]).abs()
    out["range"] = out["high"] - out["low"]
    out["body_ratio"] = out["body"] / out["range"].replace(0, np.nan)
    out["vol_ratio"] = out["volume"] / out["vol_sma"].replace(0, np.nan)

    return out


def get_prev_day_levels(df: pd.DataFrame) -> tuple[float | None, float | None]:
    """Return (prev_day_high, prev_day_low) from the most recent completed day."""
    days = sorted(df.index.normalize().unique())
    if len(days) < 2:
        return None, None
    prev_day = days[-2]
    prev_data = df[df.index.normalize() == prev_day]
    if prev_data.empty:
        return None, None
    return float(prev_data["high"].max()), float(prev_data["low"].min())


# ─── Signal Detection ─────────────────────────────────────────────────────


def check_signal(
    row, prev_row=None,
    prev_day_high=None, prev_day_low=None,
    pdl_dirs_used: set | None = None,
    atr_20: float = 0.0,
    prev_vol_r: float | None = None,
) -> tuple[str | None, str | None]:
    """Evaluate PDL + Momentum entry conditions on a single bar.

    HYBRID STRATEGY:
    - 2-Bar Volume Confirmation: requires sustained volume over 2 bars
    - Dynamic Body Ratio: 0.72 if ATR > 1.1x, else 0.70

    Returns (direction, trigger) where direction is 'LONG'/'SHORT'/None
    and trigger is 'PDL'/'MOM'/None.
    """
    c, vwap = row["close"], row["vwap"]
    rsi, vol, vsma, atr = row["rsi"], row["volume"], row["vol_sma"], row["atr"]

    if pd.isna(rsi) or pd.isna(vsma) or pd.isna(atr) or atr <= 0 or pd.isna(vwap):
        return None, None

    vol_ok = vol > vsma if vsma > 0 and not pd.isna(vsma) else False
    prev_close = prev_row["close"] if prev_row is not None else None

    # ── Entry A: Prev-Day Level Breakout ──
    if prev_day_high is not None and prev_close is not None and vol_ok:
        dirs = pdl_dirs_used or set()
        if ("LONG" not in dirs
                and prev_close <= prev_day_high and c > prev_day_high
                and c > vwap and rsi > PDL_LONG_RSI_MIN):
            return "LONG", "PDL"
        if ("SHORT" not in dirs
                and prev_close >= prev_day_low and c < prev_day_low
                and c < vwap and rsi < PDL_SHORT_RSI_MAX):
            return "SHORT", "PDL"

    # ── Entry B: Momentum Breakout (HYBRID) ──
    # Calculate ATR ratio for dynamic body ratio
    atr_ratio = atr / atr_20 if atr_20 > 0 else 1.0
    is_high_vol = atr_ratio > 1.1

    # Dynamic body ratio: tighter in high volatility
    body_threshold = 0.72 if is_high_vol else MOM_BODY_RATIO  # 0.70

    body_r = row.get("body_ratio", 0) if not pd.isna(row.get("body_ratio", float("nan"))) else 0
    vol_r = row.get("vol_ratio", 0) if not pd.isna(row.get("vol_ratio", float("nan"))) else 0

    # HYBRID: 2-bar volume confirmation (sustained volume spike)
    if not pd.isna(body_r) and not pd.isna(vol_r) and prev_vol_r is not None:
        vol_sustained = (prev_vol_r >= MOM_VOL_MULT * 0.8 and vol_r >= MOM_VOL_MULT * 0.8)
        if vol_sustained and body_r >= body_threshold:
            if (c > row["open"] and c > vwap
                    and MOM_LONG_RSI[0] <= rsi <= MOM_LONG_RSI[1]):
                return "LONG", "MOM"
            if (c < row["open"] and c < vwap
                    and MOM_SHORT_RSI[0] <= rsi <= MOM_SHORT_RSI[1]):
                return "SHORT", "MOM"

    # Fallback: single-bar momentum if no previous data
    if not pd.isna(body_r) and not pd.isna(vol_r):
        if vol_r >= MOM_VOL_MULT and body_r >= body_threshold:
            if (c > row["open"] and c > vwap
                    and MOM_LONG_RSI[0] <= rsi <= MOM_LONG_RSI[1]):
                return "LONG", "MOM"
            if (c < row["open"] and c < vwap
                    and MOM_SHORT_RSI[0] <= rsi <= MOM_SHORT_RSI[1]):
                return "SHORT", "MOM"

    return None, None


def calc_qty(capital: float, atr: float, price: float) -> int:
    risk_q = int(capital * RISK_PCT / atr)
    max_q = int(capital * LEV_CAP / price)
    return max(min(risk_q, max_q), 1)


# ─── Display ──────────────────────────────────────────────────────────────


def print_signal(
    sym, sec, direction, trigger, price, atr, qty, sl, tp,
    rsi, vwap, vol_ratio, sl_mult, ts=None,
):
    clr = GRN if direction == "LONG" else RED
    arrow = "▲" if direction == "LONG" else "▼"
    trig_lbl = "PDL" if trigger == "PDL" else "MOM"
    stamp = (ts or datetime.now()).strftime("%H:%M:%S")
    print(f"\n{'=' * 60}")
    print(f"{clr}{BLD}  {arrow} {direction} [{trig_lbl}] — {sym} ({sec})  [{stamp}]{RST}")
    print(f"{'=' * 60}")
    print(f"  Price  : ₹{price:,.2f}")
    print(f"  SL     : ₹{sl:,.2f}  ({sl_mult}× ATR = ₹{atr * sl_mult:.2f})")
    print(f"  TP     : ₹{tp:,.2f}  (RR 1:{RR})")
    print(f"  Qty    : {qty}  (risk ₹{qty * atr * sl_mult:,.0f})")
    print(f"  {'─' * 25}")
    print(f"  VWAP   : ₹{vwap:,.2f}")
    print(f"  RSI(9) : {rsi:.1f}")
    print(f"  Vol    : {vol_ratio:.1f}× avg")
    print(f"{'=' * 60}\n")


def format_tg_signal(sym, sec, direction, trigger, price, atr, qty, sl, tp, rsi, vwap, vol_ratio, sl_mult):
    icon = "\U0001f7e2" if direction == "LONG" else "\U0001f534"
    trig_lbl = "Prev-Day Break" if trigger == "PDL" else "Momentum"
    return (
        f"{icon} *{direction} [{trig_lbl}] — {sym}* ({sec})\n\n"
        f"Price: ₹{price:,.2f}\n"
        f"SL: ₹{sl:,.2f} | TP: ₹{tp:,.2f}\n"
        f"Qty: {qty} | Risk: ₹{qty * atr * sl_mult:,.0f}\n"
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
        f"  {'Symbol':<11} {'Close':>9}  {'VWAP':>9}  "
        f"{'RSI':>5}  {'Vol':>5}  {'Body':>5}  Signal"
    )
    print(f"  {'─' * 62}")
    for r in rows:
        rsi_s = f"{r['rsi']:.0f}" if not pd.isna(r.get("rsi", float("nan"))) else "  -"
        vr = r.get("vol_r", float("nan"))
        vol_s = f"{vr:.1f}x" if not pd.isna(vr) else "  -"
        br = r.get("body_r", 0)
        body_s = f"{br:.0%}" if br > 0 else "  -"

        sig = ""
        if r.get("signal"):
            c = GRN if r["signal"] == "LONG" else RED
            trig = r.get("trigger", "")
            sig = f"{c}{BLD}{r['signal']}[{trig}]{RST}"

        print(
            f"  {r['sym']:<11} ₹{r['close']:>8,.2f}  ₹{r['vwap']:>8,.2f}  "
            f"{rsi_s:>5}  {vol_s:>5}  {body_s:>5}  {sig}"
        )


# ─── Timing Helpers ────────────────────────────────────────────────────────


def in_entry_window(t: time) -> bool:
    # HYBRID: Morning-only trading (ENTRY_PM is None)
    if ENTRY_PM is None:
        return ENTRY_AM[0] <= t <= ENTRY_AM[1]
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


def fetch_prev_day_levels(data: dict[str, pd.DataFrame]) -> dict[str, tuple[float, float]]:
    """Compute previous-day high/low for each stock from warmup data."""
    levels: dict[str, tuple[float, float]] = {}
    for sym, df in data.items():
        h, l = get_prev_day_levels(df)
        if h is not None:
            levels[sym] = (h, l)
            print(f"  {sym:<11} Prev-Day  H=₹{h:>9,.2f}  L=₹{l:>9,.2f}")
            flog.info("PDL  %-11s  H=%.2f  L=%.2f", sym, h, l)
        else:
            print(f"  {YLW}{sym:<11} no prev-day data{RST}")
            flog.warning("PDL  %-11s  no prev-day data", sym)
    return levels


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
            _time.sleep(1.0)
        else:
            print(f"  {YLW}{sym:<11} no data{RST}")
            flog.warning("WARMUP  %-11s  no data (no API, no cache)", sym)
    return data


# ─── Signal Scan (shared by dry-run and live) ─────────────────────────────


def scan_bar(sym, sec, df, bar_idx, capital,
             prev_day_high=None, prev_day_low=None,
             pdl_dirs_used=None):
    """Check one bar for a signal. Returns a result dict.

    HYBRID STRATEGY:
    - ATR-based SL adjustment (dynamic position sizing)
    - 2-bar volume tracking for momentum confirmation
    """
    row = df.iloc[bar_idx]
    prev_row = df.iloc[bar_idx - 1] if bar_idx != 0 and abs(bar_idx) < len(df) else None
    ts = df.index[bar_idx]

    # Calculate ATR 20-period average for dynamic SL and body ratio
    atr_20 = df["atr"].rolling(20).mean().iloc[bar_idx] if len(df) >= 20 else df["atr"].mean()

    # Get previous bar's volume ratio for 2-bar confirmation
    prev_vol_r = None
    if prev_row is not None:
        prev_vol_r = prev_row.get("vol_ratio")
        if pd.isna(prev_vol_r):
            prev_vol_r = (
                prev_row["volume"] / prev_row["vol_sma"]
                if prev_row["vol_sma"] > 0 and not pd.isna(prev_row["vol_sma"])
                else None
            )

    sig, trigger = check_signal(
        row, prev_row, prev_day_high, prev_day_low, pdl_dirs_used,
        atr_20, prev_vol_r,
    )

    vol_r = row.get("vol_ratio", float("nan"))
    if pd.isna(vol_r):
        vol_r = (
            row["volume"] / row["vol_sma"]
            if row["vol_sma"] > 0 and not pd.isna(row["vol_sma"])
            else float("nan")
        )
    body_r = row.get("body_ratio", 0) if not pd.isna(row.get("body_ratio", float("nan"))) else 0

    result = {
        "sym": sym,
        "sec": sec,
        "close": row["close"],
        "vwap": row["vwap"],
        "rsi": row["rsi"],
        "vol_r": vol_r,
        "body_r": body_r,
        "signal": sig,
        "trigger": trigger,
        "pdl_high": prev_day_high,
        "pdl_low": prev_day_low,
    }

    if sig:
        p, a = row["close"], row["atr"]
        sl_mult = PDL_SL_MULT if trigger == "PDL" else MOM_SL_MULT

        # HYBRID: ATR-based SL adjustment
        atr_ratio = a / atr_20 if atr_20 > 0 else 1.0
        if atr_ratio > 1.3:
            sl_mult *= 1.2  # Wider SL in high volatility
        elif atr_ratio < 0.7:
            sl_mult *= 0.9  # Tighter SL in calm conditions

        sl_dist = a * sl_mult
        q = calc_qty(capital, sl_dist, p)
        sl = p - sl_dist if sig == "LONG" else p + sl_dist
        tp = p + sl_dist * RR if sig == "LONG" else p - sl_dist * RR
        vr = vol_r if not pd.isna(vol_r) else 0
        result.update(price=p, atr=a, qty=q, sl=sl, tp=tp,
                      vol_ratio=vr, ts=ts, sl_mult=sl_mult)

    return result


# ─── Fetch helper for live loop ───────────────────────────────────────────


def _fetch_latest(client, sym, stock_data, now):
    """Fetch the latest few candles and merge into existing data.

    Drops the current incomplete candle so indicators only use closed bars.
    """
    window_start = (now - timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M")
    window_end = (now - timedelta(minutes=3)).strftime("%Y-%m-%d %H:%M")
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

    print(f"\n{CYN}[pdl]{RST} Computing prev-day levels...")
    pdl_levels = fetch_prev_day_levels(data)

    ref_sym = next(iter(data))
    scan_date = data[ref_sym].index[-1].date()
    print(f"\n{YLW}{BLD}── DRY RUN: Scanning {scan_date} for signals ──{RST}\n")

    signals_found = 0
    stag_exits = 0
    cooldowns: dict[str, datetime] = {}
    daily_sl_counts: dict[str, int] = {}
    for sym, sec in STOCKS:
        if sym not in data:
            continue
        df = compute_indicators(data[sym])
        pdl_h, pdl_l = pdl_levels.get(sym, (None, None))
        pdl_dirs_used: set[str] = set()

        current_pos: dict | None = None
        bars_since_entry = 0

        day_mask = df.index.date == scan_date
        for i in df.index[day_mask]:
            idx = df.index.get_loc(i)

            if current_pos is not None:
                bars_since_entry += 1
                row = df.iloc[idx]
                hit = _check_bracket_hit(current_pos, row["high"], row["low"])
                if hit:
                    if hit == "SL":
                        daily_sl_counts[sym] = daily_sl_counts.get(sym, 0) + 1
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
            if daily_sl_counts.get(sym, 0) >= MAX_SL_PER_DAY:
                continue

            result = scan_bar(sym, sec, df, idx, capital, pdl_h, pdl_l, pdl_dirs_used)
            if result["signal"]:
                signals_found += 1
                cooldowns[sym] = i + COOLDOWN
                if result["trigger"] == "PDL":
                    pdl_dirs_used.add(result["signal"])
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
                    sym, sec, r["signal"], r["trigger"], r["price"], r["atr"],
                    r["qty"], r["sl"], r["tp"], r["rsi"], r["vwap"],
                    r["vol_ratio"], r.get("sl_mult", 1.0), ts=r["ts"],
                )
                phase = get_phase(i.time())
                log_signal(
                    sym, sec, r["signal"], r["trigger"], r["price"],
                    r["sl"], r["tp"], r["atr"], r["qty"], r["rsi"],
                    r["vwap"], r["vol_ratio"], ts=i, phase=phase,
                )

    log_file = LOG_DIR / f"signals_{scan_date}.csv"
    print(f"\n{CYN}── Dry run complete: {signals_found} signal(s), {stag_exits} stagnation exit(s) on {scan_date} ──{RST}")
    if signals_found:
        print(f"{DIM}  Log saved: {log_file}{RST}")
    if stag_exits:
        print(f"{DIM}  Stagnation exit: {STAG_MAX_BARS} bars, < {STAG_MIN_PROFIT_ATR}× ATR{RST}")


# ─── Live Loop (Phase-Based Architecture) ─────────────────────────────────


def live(capital: float) -> None:
    flog.info("=" * 70)
    flog.info("SESSION START  Capital=%.0f  RR=1:%.1f  Risk=%.0f%%", capital, RR, RISK_PCT * 100)
    flog.info("Stocks: %s", ", ".join(s for s, _ in STOCKS))
    flog.info("Strategy: PDL (SL=%.1fx ATR) + MOM (SL=%.1fx ATR)  MaxSL/day=%d", PDL_SL_MULT, MOM_SL_MULT, MAX_SL_PER_DAY)

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

    print(f"\n{CYN}[data] Fetching {WARMUP_3MIN_DAYS}-day 3-min candles (indicator warmup){RST}")
    flog.info("Loading %d-day 3-min warmup data", WARMUP_3MIN_DAYS)
    stock_data = load_warmup(client)
    loaded = len(stock_data)

    if loaded == 0:
        print(f"{RED}No data loaded. Exiting.{RST}")
        flog.error("No data loaded — exiting")
        return

    print(f"\n{CYN}[pdl] Computing prev-day breakout levels{RST}")
    pdl_levels = fetch_prev_day_levels(stock_data)

    syms = ", ".join(s for s, _ in STOCKS if s in stock_data)

    flog.info("Warm-up complete: %d stocks loaded, PDL levels for %d", loaded, len(pdl_levels))

    print(f"\n{BLD}{'═' * 60}")
    print(f"  LIVE SIGNAL MONITOR — PDL + Momentum")
    print(f"  Capital : ₹{capital:,.0f}  |  RR 1:{RR}  |  Risk {RISK_PCT*100:.0f}%")
    print(f"  Stocks  : {loaded} loaded  |  {len(pdl_levels)} with PDL levels")
    print(f"  PDL SL  : {PDL_SL_MULT}× ATR  |  MOM SL : {MOM_SL_MULT}× ATR")
    print(f"  {'─' * 56}")
    print(f"  09:14  Phase 1  Warm-Up       ✓ done")
    print(f"  09:15  Phase 2  Silent         track only, no signals")
    print(f"  10:00  Phase 3  Morning        signals ON")
    print(f"  12:00  Phase 4  Midday         track only, no signals")
    print(f"  14:00  Phase 5  Afternoon      signals ON")
    print(f"  15:00  Phase 6  Shutdown       stop")
    print(f"{'═' * 60}{RST}\n")

    send_telegram(
        f"\U0001f514 *Signal Monitor Started — PDL + Momentum*\n"
        f"Capital: ₹{capital:,.0f}\n"
        f"Stocks: {syms}\n"
        f"PDL SL: {PDL_SL_MULT}× ATR | MOM SL: {MOM_SL_MULT}× ATR\n"
        f"Schedule: 10:00–12:00, 14:00–15:00"
    )

    cooldowns: dict[str, datetime] = {}
    open_positions: dict[str, dict] = {}
    signal_counts: dict[tuple[str, str], int] = {}
    pdl_dirs_used: dict[str, set[str]] = {}
    daily_sl_counts: dict[str, int] = {}
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
                pdl_dirs_used.clear()
                daily_sl_counts.clear()
                today_signals.clear()
                today_exits.clear()
                scan_count = 0
                last_scan_candle = -1
                last_phase = ""
                print(f"\n{CYN}[new day] {current_date}{RST}")
                print(f"{CYN}[pdl] Refreshing prev-day levels...{RST}")
                pdl_levels = fetch_prev_day_levels(stock_data)

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
                wait = max(int((datetime.combine(now.date(), MKT_OPEN) - now).total_seconds()), 0)
                print(f"{DIM}[{now:%H:%M}] Pre-market. Opens in {wait // 60}m {wait % 60}s...{RST}")
                _time.sleep(max(min(wait, 60), 1))
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
                            _time.sleep(1.0)
                            continue
                    elif "rate" in str(exc).lower() or "access denied" in str(exc).lower():
                        _time.sleep(2.0)
                        try:
                            _fetch_latest(client, sym, stock_data, now)
                        except Exception:
                            flog.warning("API  %-11s  rate-limit retry failed", sym)
                            continue
                    else:
                        print(f"  {DIM}{sym}: {exc}{RST}")
                        flog.warning("API  %-11s  fetch error: %s", sym, exc)
                        _time.sleep(1.0)
                        continue

                _time.sleep(1.0)

                df = compute_indicators(stock_data[sym])
                pdl_h, pdl_l = pdl_levels.get(sym, (None, None))
                sym_pdl_dirs = pdl_dirs_used.setdefault(sym, set())
                last_row = df.iloc[-1]

                if sym in open_positions:
                    pos = open_positions[sym]
                    hit = _check_bracket_hit(pos, last_row["high"], last_row["low"])
                    if hit:
                        flog.info("BRACKET  %-11s  %s hit — position closed", sym, hit)
                        if hit == "SL":
                            daily_sl_counts[sym] = daily_sl_counts.get(sym, 0) + 1
                            flog.info("SL_COUNT  %-11s  %d / %d", sym, daily_sl_counts[sym], MAX_SL_PER_DAY)
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
                    sym, sec, df, -1, capital, pdl_h, pdl_l, sym_pdl_dirs,
                )

                on_cooldown = sym in cooldowns and now < cooldowns[sym]

                r = result
                rsi_str = f"{r['rsi']:.0f}" if not pd.isna(r.get("rsi", float("nan"))) else "-"
                vr = r.get("vol_r", float("nan"))
                vol_str = f"{vr:.1f}x" if not pd.isna(vr) else "-"
                raw_sig = r.get("signal") or "-"
                raw_trig = r.get("trigger") or "-"

                already_in_position = sym in open_positions
                session_key = "AM" if phase == "ACTIVE_AM" else "PM"
                at_signal_cap = signal_counts.get((sym, session_key), 0) >= MAX_SIGNALS_PER_STOCK
                at_sl_cap = daily_sl_counts.get(sym, 0) >= MAX_SL_PER_DAY
                if is_active and result["signal"] and not on_cooldown and not already_in_position and not at_signal_cap and not at_sl_cap:
                    print_signal(
                        sym, sec, r["signal"], r["trigger"], r["price"],
                        r["atr"], r["qty"], r["sl"], r["tp"], r["rsi"],
                        r["vwap"], r["vol_ratio"], r.get("sl_mult", 1.0),
                    )
                    send_telegram(format_tg_signal(
                        sym, sec, r["signal"], r["trigger"], r["price"],
                        r["atr"], r["qty"], r["sl"], r["tp"], r["rsi"],
                        r["vwap"], r["vol_ratio"], r.get("sl_mult", 1.0),
                    ))
                    log_signal(
                        sym, sec, r["signal"], r["trigger"], r["price"],
                        r["sl"], r["tp"], r["atr"], r["qty"], r["rsi"],
                        r["vwap"], r["vol_ratio"], phase=phase,
                    )
                    cooldowns[sym] = now + COOLDOWN
                    signal_counts[(sym, session_key)] = signal_counts.get((sym, session_key), 0) + 1
                    if r["trigger"] == "PDL":
                        sym_pdl_dirs.add(r["signal"])
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
                        "trigger": r["trigger"],
                        "price": r["price"], "sl": r["sl"], "tp": r["tp"],
                        "time": now,
                    })
                    flog.info(
                        "  %-11s  C=%.2f  VWAP=%.2f  RSI=%s  Vol=%s  → ★ %s[%s]  SL=%.2f  TP=%.2f  Qty=%d",
                        sym, r["close"], r["vwap"], rsi_str,
                        vol_str, r["signal"], r["trigger"], r["sl"], r["tp"], r["qty"],
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
                    elif at_sl_cap:
                        reason = f"sl_cap({daily_sl_counts.get(sym, 0)}/{MAX_SL_PER_DAY})"
                    elif raw_sig != "-":
                        reason = "blocked"
                    result["signal"] = None
                    flog.debug(
                        "  %-11s  C=%.2f  VWAP=%.2f  RSI=%s  Vol=%s  → %s[%s]  %s",
                        sym, r["close"], r["vwap"], rsi_str,
                        vol_str, raw_sig, raw_trig, reason,
                    )

                dashboard.append(result)

            print_dashboard(dashboard, phase)

    except KeyboardInterrupt:
        n = len(today_signals)
        flog.info("STOPPED MANUALLY  %d signal(s), %d scans", n, scan_count)
        for s in today_signals:
            flog.info(
                "  SUMMARY  %s  %5s  %-11s  @ %.2f  [%s]",
                s["time"].strftime("%H:%M"), s["dir"], s["sym"],
                s["price"], s.get("trigger", "?"),
            )
        print(f"\n\n{YLW}[stopped] {n} signal(s) generated today:{RST}")
        for s in today_signals:
            trig = s.get("trigger", "?")
            print(f"  {s['time']:%H:%M}  {s['dir']:>5}  [{trig:>3}]  {s['sym']:<11}  @ ₹{s['price']:,.2f}")
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
    print(f"{DIM}PDL (SL {PDL_SL_MULT}×ATR) + Momentum (SL {MOM_SL_MULT}×ATR) | VWAP + RSI + Vol")
    print(f"RR 1:{RR} | Risk {RISK_PCT * 100:.0f}% | Max {MAX_SL_PER_DAY} SL/stock/day | Polls at HH:MM:01{RST}\n")

    if args.dry_run:
        dry_run(args.capital)
    else:
        live(args.capital)


if __name__ == "__main__":
    main()

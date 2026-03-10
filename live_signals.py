"""
Live Signal Generator for NSE Intraday Scalping Strategy — PREARMED PDL

Streams live data via Angel One WebSocket, computes indicators in
real-time (pandas/numpy), and sends BUY/SHORT alerts to console + Telegram.

STRATEGY:
  Prearmed PDL — park a breakout trigger just beyond yesterday's High/Low.
  Signal fires when the completed 3-minute bar trades through that trigger
  and still closes through the base prev-day level.

Daily Lifecycle:
  Phase 1  09:14  Warm-Up   — connect, fetch 5 days of 3-min data
  Phase 2  09:15  Silent    — track indicators every 3 min, NO signals
  Phase 3  09:30  Active    — trading window, signals enabled
  Phase 4  13:00  Shutdown  — stop signals, track only until close

Entry   : Prearmed PDL trigger = prev-day High/Low ± 0.1× ATR
Exit    : SL = 1.0× ATR, TP = 2.5R, max 1 signal per direction/day
Sizing  : 1.5% risk per trade, 5× leverage cap
Guard   : Max 2 SL hits per stock per day — blocks further entries

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
import threading
import time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from event_calendar import get_excluded_stocks
from fetch_data import AngelOneClient, load_csv
from websocket_feed import WebSocketFeed

load_dotenv()


# ─── Stock Configuration ──────────────────────────────────────────────────

STOCKS = [
    ("JSWSTEEL", "Steel"),
    ("HINDUNILVR", "FMCG"),
    ("AXISBANK", "Banking"),
    ("TATASTEEL", "Steel"),
    ("TRENT", "Retail"),
    ("BAJAJ-AUTO", "Auto"),
    ("BAJFINANCE", "Finance"),
    ("HDFCBANK", "Banking"),
    ("POWERGRID", "Power"),
    ("HCLTECH", "IT"),
]

# ─── Strategy Parameters ──────────────────────────────────────────────────

RSI_PERIOD = 9
VOL_SMA = 20
ATR_PERIOD = 14

RR = 2.5
RISK_PCT = 0.015
LEV_CAP = 5.0

PDL_PREARM_BUFFER_ATR = 0.10
PDL_SL_MULT = 1.0
MAX_SL_PER_DAY = 2

SIGNAL_VARIANTS = [
    {"key": "baseline", "label": "Baseline", "body_close_min": None, "trigger_name": "PDL_BASE"},
    {"key": "bodyclose50", "label": "BodyClose50", "body_close_min": 0.50, "trigger_name": "PDL_BC50"},
]


# ─── Time Rules & Phases ──────────────────────────────────────────────────

MKT_OPEN = time(9, 15)
ENTRY_AM = (time(9, 30), time(13, 0))
ENTRY_PM = None
MKT_CLOSE = time(15, 0)

BAR_INTERVAL = timedelta(minutes=3)
COOLDOWN = timedelta(minutes=30)
MAX_SIGNALS_PER_STOCK = 2
WARMUP_3MIN_DAYS = 5
POLL_BUFFER_SEC = 1
WS_STALE_SEC = 20

DATA_DIR = Path(__file__).parent / "data"
LOG_DIR = Path(__file__).parent / "logs"
LIVE_LOG_DIR = LOG_DIR / "live_signals"
SIGNALS_DIR = LOG_DIR / "signals"

logging.getLogger().handlers = []
logging.getLogger().addHandler(logging.NullHandler())

for _lib_name in ("smartConnect", "SmartApi", "logzero", "SmartWebSocketV2",
                   "websocket", "urllib3"):
    _lib_log = logging.getLogger(_lib_name)
    _lib_log.setLevel(logging.CRITICAL)
    _lib_log.handlers = []
    _lib_log.addHandler(logging.NullHandler())


def _init_file_logger() -> logging.Logger:
    """Return the live_signals logger. File handler added later by _attach_log_file()."""
    logger = logging.getLogger("live_signals")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger


def _attach_log_file():
    """Create the dated log file. Call once after any overnight wait loop."""
    LIVE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    flog.handlers = [h for h in flog.handlers if not isinstance(h, logging.FileHandler)]
    fh = logging.FileHandler(
        LIVE_LOG_DIR / f"live_{datetime.now():%Y-%m-%d}.log", encoding="utf-8",
    )
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
    flog.addHandler(fh)


flog = _init_file_logger()

_LOG_COLUMNS = [
    "date", "time", "symbol", "sector", "direction", "trigger", "variant", "price",
    "sl", "tp", "atr", "qty", "risk", "vwap", "rsi", "vol_ratio", "phase",
]

PHASES = {
    "PRE_MARKET": "Pre-market",
    "SILENT":     "Phase 2 · Silent tracking",
    "ACTIVE_AM":  "Phase 3 · Active trading",
    "SHUTDOWN":   "Phase 4 · Shutdown",
}


def get_phase(t: time) -> str:
    if t < MKT_OPEN:
        return "PRE_MARKET"
    if t < ENTRY_AM[0]:
        return "SILENT"
    if t <= ENTRY_AM[1]:
        return "ACTIVE_AM"
    if ENTRY_PM is None:
        return "SHUTDOWN"
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
    sym, sec, direction, trigger, variant, price, sl, tp, atr, qty, rsi, vwap,
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
        "variant": variant or "",
        "price": round(price, 2),
        "sl": round(sl, 2),
        "tp": round(tp, 2),
        "atr": round(atr, 2),
        "qty": qty,
        "risk": round(qty * abs(price - sl), 2),
        "vwap": round(vwap, 2),
        "rsi": round(rsi, 1),
        "vol_ratio": round(vol_ratio, 1),
        "phase": phase,
    }

    if log_file.exists():
        with open(log_file, newline="") as f:
            raw_rows = list(csv.reader(f))
            existing_columns = raw_rows[0] if raw_rows else []
            if existing_columns != _LOG_COLUMNS:
                migrated_rows = []
                old_columns = [c for c in _LOG_COLUMNS if c != "variant"]
                for values in raw_rows[1:]:
                    if not values:
                        continue
                    if len(values) == len(_LOG_COLUMNS):
                        migrated_rows.append(dict(zip(_LOG_COLUMNS, values)))
                        continue
                    if len(values) == len(old_columns):
                        old = dict(zip(old_columns, values))
                        migrated = {
                            "date": old.get("date", ""),
                            "time": old.get("time", ""),
                            "symbol": old.get("symbol", ""),
                            "sector": old.get("sector", ""),
                            "direction": old.get("direction", ""),
                            "trigger": old.get("trigger", ""),
                            "variant": "",
                            "price": old.get("price", ""),
                            "sl": old.get("sl", ""),
                            "tp": old.get("tp", ""),
                            "atr": old.get("atr", ""),
                            "qty": old.get("qty", ""),
                            "risk": old.get("risk", ""),
                            "vwap": old.get("vwap", ""),
                            "rsi": old.get("rsi", ""),
                            "vol_ratio": old.get("vol_ratio", ""),
                            "phase": old.get("phase", ""),
                        }
                        migrated_rows.append(migrated)
                with open(log_file, "w", newline="") as wf:
                    writer = csv.DictWriter(wf, fieldnames=_LOG_COLUMNS)
                    writer.writeheader()
                    writer.writerows(migrated_rows)
                is_new = False

    with open(log_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_LOG_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def _trigger_label(trigger: str) -> str:
    return {
        "PDL_BASE": "Prearmed PDL",
        "PDL_BC50": "Prearmed PDL + BodyClose50",
    }.get(trigger, trigger)


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


def get_prev_day_levels(
    df: pd.DataFrame, session_date: date,
) -> tuple[float | None, float | None]:
    """Return the previous trading day's high/low for a session date."""
    if df.empty:
        return None, None

    ref_day = pd.Timestamp(session_date).normalize()
    prior_days = sorted(day for day in df.index.normalize().unique() if day < ref_day)
    if not prior_days:
        return None, None

    prev_day = prior_days[-1]
    prev_data = df[df.index.normalize() == prev_day]
    if prev_data.empty:
        return None, None
    return float(prev_data["high"].max()), float(prev_data["low"].min())


# ─── Signal Detection ─────────────────────────────────────────────────────


def check_signal(
    row, prev_row=None,
    prev_day_high=None, prev_day_low=None,
    pdl_dirs_used: set | None = None,
    body_close_min: float | None = None,
    trigger_name: str = "PDL_BASE",
    **_kwargs,
) -> tuple[str | None, str | None]:
    """Evaluate the prearmed PDL trigger on a completed bar.

    Returns (direction, trigger) where direction is 'LONG'/'SHORT'/None
    and trigger is the configured variant trigger name or None.
    """
    c, h, l = row["close"], row["high"], row["low"]
    atr = row["atr"]

    if pd.isna(atr) or atr <= 0:
        return None, None

    prev_close = prev_row["close"] if prev_row is not None else None
    if prev_close is None:
        return None, None

    dirs = pdl_dirs_used or set()
    buffer = atr * PDL_PREARM_BUFFER_ATR
    def body_close_ok(direction: str) -> bool:
        if body_close_min is None:
            return True
        bar_range = h - l
        if pd.isna(bar_range) or bar_range <= 0:
            return False
        if direction == "LONG":
            close_pos = (c - l) / bar_range
        else:
            close_pos = (h - c) / bar_range
        return close_pos >= body_close_min

    if prev_day_high is not None:
        long_trigger = prev_day_high + buffer
        if ("LONG" not in dirs
                and prev_close <= prev_day_high
                and h >= long_trigger
                and c >= prev_day_high
                and body_close_ok("LONG")):
            return "LONG", trigger_name
    if prev_day_low is not None:
        short_trigger = prev_day_low - buffer
        if ("SHORT" not in dirs
                and prev_close >= prev_day_low
                and l <= short_trigger
                and c <= prev_day_low
                and body_close_ok("SHORT")):
            return "SHORT", trigger_name

    return None, None


def calc_qty(capital: float, atr: float, price: float) -> int:
    risk_q = int(capital * RISK_PCT / atr)
    max_q = int(capital * LEV_CAP / price)
    return max(min(risk_q, max_q), 1)


# ─── Display ──────────────────────────────────────────────────────────────


def print_signal(
    sym, sec, direction, trigger, price, atr, qty, sl, tp,
    rsi, vwap, vol_ratio, sl_mult, ts=None, variant="",
):
    clr = GRN if direction == "LONG" else RED
    arrow = "▲" if direction == "LONG" else "▼"
    trig_lbl = _trigger_label(trigger)
    stamp = (ts or datetime.now()).strftime("%H:%M:%S")
    print(f"\n{'=' * 60}")
    print(f"{clr}{BLD}  {arrow} {direction} [{trig_lbl}] — {sym} ({sec}) · {variant}  [{stamp}]{RST}")
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


def format_tg_signal(sym, sec, direction, trigger, variant, price, atr, qty, sl, tp, rsi, vwap, vol_ratio, sl_mult):
    icon = "\U0001f7e2" if direction == "LONG" else "\U0001f534"
    trig_lbl = _trigger_label(trigger)
    return (
        f"{icon} *{direction} [{trig_lbl}] — {sym}* ({sec})\n"
        f"Variant: {variant}\n\n"
        f"Price: ₹{price:,.2f}\n"
        f"SL: ₹{sl:,.2f} | TP: ₹{tp:,.2f}\n"
        f"Qty: {qty} | Risk: ₹{qty * atr * sl_mult:,.0f}\n"
        f"RR: 1:{RR}\n\n"
        f"VWAP: ₹{vwap:,.2f} | RSI: {rsi:.1f}\n"
        f"Vol: {vol_ratio:.1f}× avg"
    )


# ─── Bracket Exit ─────────────────────────────────────────────────────────


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
    # Morning-only trading by default (ENTRY_PM is None)
    if ENTRY_PM is None:
        return ENTRY_AM[0] <= t <= ENTRY_AM[1]
    return (ENTRY_AM[0] <= t <= ENTRY_AM[1]) or (ENTRY_PM[0] <= t <= ENTRY_PM[1])


def bar_completed_at(ts: pd.Timestamp) -> pd.Timestamp:
    """Return the completed-at timestamp for a stored 3-minute bar.

    WebSocket-built candles are keyed by candle open time, so adding one
    bar interval yields the effective signal time.
    """
    return ts + BAR_INTERVAL


# ─── Data Loading ─────────────────────────────────────────────────────────

_FS_SAFE = str.maketrans({"&": ""})


def _cache_path(symbol: str) -> Path:
    return DATA_DIR / f"{symbol.translate(_FS_SAFE)}_3min.csv"


def fetch_prev_day_levels(
    data: dict[str, pd.DataFrame], session_date: date,
) -> dict[str, tuple[float, float]]:
    """Compute previous-day high/low for each stock for a session date."""
    levels: dict[str, tuple[float, float]] = {}
    for sym, df in data.items():
        h, l = get_prev_day_levels(df, session_date)
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
            fetched = False
            for attempt in range(3):
                try:
                    df = client.fetch_history(sym, days=WARMUP_3MIN_DAYS)
                    data[sym] = df
                    DATA_DIR.mkdir(parents=True, exist_ok=True)
                    df.to_csv(cache)
                    print(
                        f"  {sym:<11} {len(df):>6} bars  "
                        f"(API: {df.index[0]:%m-%d} → {df.index[-1]:%m-%d %H:%M})"
                    )
                    flog.info(
                        "WARMUP  %-11s  %d bars  API %s→%s  cached",
                        sym, len(df), df.index[0].date(), df.index[-1],
                    )
                    fetched = True
                    break
                except Exception as exc:
                    if attempt < 2:
                        flog.warning("WARMUP  %-11s  attempt %d failed: %s — retrying in %ds", sym, attempt + 1, exc, 3 * (attempt + 1))
                        _time.sleep(3.0 * (attempt + 1))
                    else:
                        if cache.exists():
                            try:
                                df = load_csv(str(cache))
                                data[sym] = df
                                print(f"  {YLW}{sym:<11} {len(df):>6} bars  (stale cache: {df.index[-1]:%Y-%m-%d}){RST}")
                                flog.warning("WARMUP  %-11s  %d bars  stale_cache=%s (API: %s)", sym, len(df), df.index[-1].date(), exc)
                                fetched = True
                            except Exception:
                                pass
                        if not fetched:
                            print(f"  {RED}{sym:<11} FAILED: {exc}{RST}")
                            flog.error("WARMUP  %-11s  FAILED after 3 attempts: %s", sym, exc)
            _time.sleep(1.5)
        else:
            print(f"  {YLW}{sym:<11} no data{RST}")
            flog.warning("WARMUP  %-11s  no data (no API, no cache)", sym)
    return data


# ─── Signal Scan (shared by dry-run and live) ─────────────────────────────


def scan_bar(sym, sec, df, bar_idx, capital,
             prev_day_high=None, prev_day_low=None,
             pdl_dirs_used=None, variant_label="Baseline",
             body_close_min: float | None = None, trigger_name: str = "PDL_BASE"):
    """Check one bar for a prearmed PDL signal. Returns a result dict."""
    row = df.iloc[bar_idx]
    prev_row = df.iloc[bar_idx - 1] if bar_idx != 0 and abs(bar_idx) < len(df) else None
    ts = df.index[bar_idx]

    sig, trigger = check_signal(
        row, prev_row, prev_day_high, prev_day_low, pdl_dirs_used,
        body_close_min=body_close_min, trigger_name=trigger_name,
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
        "variant": variant_label,
        "pdl_high": prev_day_high,
        "pdl_low": prev_day_low,
    }

    if sig:
        a = row["atr"]
        p = (
            prev_day_high + a * PDL_PREARM_BUFFER_ATR
            if sig == "LONG" else prev_day_low - a * PDL_PREARM_BUFFER_ATR
        )
        sl_dist = a * PDL_SL_MULT
        q = calc_qty(capital, sl_dist, p)
        sl = p - sl_dist if sig == "LONG" else p + sl_dist
        tp = p + sl_dist * RR if sig == "LONG" else p - sl_dist * RR
        vr = vol_r if not pd.isna(vol_r) else 0
        result.update(price=p, atr=a, qty=q, sl=sl, tp=tp,
                      vol_ratio=vr, ts=ts, sl_mult=PDL_SL_MULT)

    return result


# ─── Fetch helper for live loop ───────────────────────────────────────────


_api_lock = threading.Lock()
_next_api_allowed = 0.0


def _rate_limit_wait():
    """Enforce 3 requests per second for Angel One getCandleData endpoint."""
    global _next_api_allowed
    with _api_lock:
        now_ts = _time.time()
        if now_ts < _next_api_allowed:
            _time.sleep(_next_api_allowed - now_ts)
        _next_api_allowed = _time.time() + 0.35


def _fetch_latest(client, sym, stock_data, now, retries=1):
    """Fetch candles since the last known bar and merge into existing data.

    Dynamically extends the lookback window to cover any gaps from missed scans.
    Drops the current incomplete candle so indicators only use closed bars.
    """
    last_bar_ts = stock_data[sym].index[-1] if sym in stock_data and not stock_data[sym].empty else None
    if last_bar_ts:
        gap_mins = max(10, int((now - last_bar_ts).total_seconds() / 60) + 5)
        lookback = min(gap_mins, 120)
    else:
        lookback = 10

    window_start = (now - timedelta(minutes=lookback)).strftime("%Y-%m-%d %H:%M")
    window_end = (now - timedelta(minutes=3)).strftime("%Y-%m-%d %H:%M")
    fresh = client.fetch_candles(sym, window_start, window_end, retries=retries)

    base = now.replace(hour=9, minute=15, second=0, microsecond=0)
    elapsed = (now - base).total_seconds()
    if elapsed >= 0:
        current_candle_open = base + timedelta(seconds=int(elapsed / 180) * 180)
        fresh = fresh[fresh.index < current_candle_open]

    if fresh.empty:
        return

    combined = pd.concat([stock_data[sym], fresh])
    stock_data[sym] = combined[~combined.index.duplicated(keep="last")].sort_index()


def _fetch_stock_threaded(client, sym, stock_data, now):
    """Rate-limited fetch for one stock, designed for ThreadPoolExecutor."""
    _rate_limit_wait()
    try:
        _fetch_latest(client, sym, stock_data, now, retries=1)
        return (sym, True, None)
    except Exception as exc:
        return (sym, False, exc)


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

    ref_sym = next(iter(data))
    scan_date = data[ref_sym].index[-1].date()
    print(f"\n{CYN}[pdl]{RST} Computing prev-day levels...")
    pdl_levels = fetch_prev_day_levels(data, scan_date)
    print(f"\n{YLW}{BLD}── DRY RUN: Scanning {scan_date} for signals ──{RST}\n")

    signals_found = 0
    cooldowns: dict[tuple[str, str], datetime] = {}
    daily_sl_counts: dict[tuple[str, str], int] = {}
    for sym, sec in STOCKS:
        if sym not in data:
            continue
        df = compute_indicators(data[sym])
        pdl_h, pdl_l = pdl_levels.get(sym, (None, None))
        pdl_dirs_used: dict[tuple[str, str], set] = {
            (cfg["key"], sym): set() for cfg in SIGNAL_VARIANTS
        }
        current_positions: dict[tuple[str, str], dict] = {}

        day_mask = df.index.date == scan_date
        for i in df.index[day_mask]:
            idx = df.index.get_loc(i)
            row = df.iloc[idx]

            for cfg in SIGNAL_VARIANTS:
                state_key = (cfg["key"], sym)
                current_pos = current_positions.get(state_key)
                if current_pos is not None:
                    hit = _check_bracket_hit(current_pos, row["high"], row["low"])
                    if hit:
                        if hit == "SL":
                            daily_sl_counts[state_key] = daily_sl_counts.get(state_key, 0) + 1
                        del current_positions[state_key]
                    continue

                if not in_entry_window(i.time()):
                    continue
                on_cooldown = state_key in cooldowns and i < cooldowns[state_key]
                if on_cooldown:
                    continue
                if daily_sl_counts.get(state_key, 0) >= MAX_SL_PER_DAY:
                    continue

                result = scan_bar(
                    sym, sec, df, idx, capital, pdl_h, pdl_l,
                    pdl_dirs_used[state_key],
                    variant_label=cfg["label"],
                    body_close_min=cfg["body_close_min"],
                    trigger_name=cfg["trigger_name"],
                )
                if result["signal"]:
                    signals_found += 1
                    cooldowns[state_key] = i + COOLDOWN
                    pdl_dirs_used[state_key].add(result["signal"])
                    current_positions[state_key] = {
                        "direction": result["signal"],
                        "entry_price": result["price"],
                        "entry_atr": result["atr"],
                        "sl": result["sl"],
                        "tp": result["tp"],
                    }
                    r = result
                    print_signal(
                        sym, sec, r["signal"], r["trigger"], r["price"], r["atr"],
                        r["qty"], r["sl"], r["tp"], r["rsi"], r["vwap"],
                        r["vol_ratio"], r.get("sl_mult", 1.0), ts=r["ts"],
                        variant=r["variant"],
                    )
                    phase = get_phase(i.time())
                    log_signal(
                        sym, sec, r["signal"], r["trigger"], r["variant"], r["price"],
                        r["sl"], r["tp"], r["atr"], r["qty"], r["rsi"],
                        r["vwap"], r["vol_ratio"], ts=i, phase=phase,
                    )

    log_file = SIGNALS_DIR / f"signals_{scan_date}.csv"
    print(f"\n{CYN}── Dry run complete: {signals_found} signal(s) on {scan_date} ──{RST}")
    if signals_found:
        print(f"{DIM}  Log saved: {log_file}{RST}")


# ─── Live Loop (Phase-Based Architecture) ─────────────────────────────────


def live(capital: float, use_websocket: bool = True) -> None:
    now_t = datetime.now().time()
    shutdown_at = ENTRY_AM[1] if ENTRY_PM is None else MKT_CLOSE
    if now_t > shutdown_at:
        window = f"{ENTRY_AM[0]:%H:%M}–{shutdown_at:%H:%M}"
        print(f"\n{YLW}{BLD}Market session is over (current time: {now_t:%H:%M}).{RST}")
        print(f"{YLW}Trading window is {window}.{RST}")
        print(f"{YLW}Waiting for next trading day (warmup starts at 09:00)...{RST}\n")
        while True:
            now = datetime.now()
            if now.weekday() < 5 and time(9, 0) <= now.time() < shutdown_at:
                break
            _time.sleep(60)
        print(f"{GRN}{BLD}Market day detected — resuming startup.{RST}\n")

    _attach_log_file()

    flog.info("=" * 70)
    flog.info("SESSION START  Capital=%.0f  RR=1:%.1f  Risk=%.1f%%  ws=%s",
              capital, RR, RISK_PCT * 100, use_websocket)
    flog.info("Stocks: %s", ", ".join(s for s, _ in STOCKS))
    flog.info("Strategy: PDL-Only (SL=%.1fx ATR)  RR=1:%.1f  MaxSL/day=%d", PDL_SL_MULT, RR, MAX_SL_PER_DAY)

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

    # ── Phase 1a: Event Calendar ─────────────────────────────────────
    print(f"\n{CYN}[cal] Checking NSE event calendar...{RST}")
    flog.info("Checking event calendar for corporate events")
    excluded_stocks: dict[str, str] = {}
    try:
        all_syms = [s for s, _ in STOCKS]
        excluded_stocks = get_excluded_stocks(all_syms)
        if excluded_stocks:
            for sym, reason in excluded_stocks.items():
                print(f"  {YLW}⚠ {sym:<14} SKIPPED — {reason}{RST}")
                flog.warning("EVENT SKIP  %-14s  %s", sym, reason)
                stock_data.pop(sym, None)
        else:
            print(f"  {GRN}No corporate events — all stocks cleared{RST}")
            flog.info("Event calendar: all stocks cleared")
    except Exception as e:
        print(f"  {YLW}Event calendar check failed ({e}) — trading all stocks{RST}")
        flog.warning("Event calendar failed: %s — no stocks excluded", e)

    print(f"\n{CYN}[pdl] Computing prev-day breakout levels{RST}")
    pdl_levels = fetch_prev_day_levels(stock_data, datetime.now().date())

    active_stocks = [(s, sec) for s, sec in STOCKS if s in stock_data]
    syms = ", ".join(s for s, _ in active_stocks)

    flog.info("Warm-up complete: %d stocks loaded (%d excluded), PDL levels for %d",
              loaded, len(excluded_stocks), len(pdl_levels))

    print(f"\n{BLD}{'═' * 60}")
    print(f"  LIVE SIGNAL MONITOR — PDL Breakout")
    print(f"  Capital : ₹{capital:,.0f}  |  RR 1:{RR}  |  Risk {RISK_PCT*100:.1f}%")
    print(f"  Stocks  : {len(active_stocks)} active  |  {len(pdl_levels)} with PDL levels")
    if excluded_stocks:
        print(f"  Skipped : {', '.join(excluded_stocks.keys())} (events)")
    print(f"  SL      : {PDL_SL_MULT}× ATR  |  Cooldown : {int(COOLDOWN.total_seconds()//60)} min")
    print(f"  {'─' * 56}")
    print(f"  09:14  Phase 1  Warm-Up       ✓ done")
    print(f"  09:15  Phase 2  Silent         track only, no signals")
    print(f"  09:30  Phase 3  Active         signals ON")
    print(f"  13:00  Phase 4  Shutdown       stop")
    print(f"{'═' * 60}{RST}\n")

    # ── Phase 1b: WebSocket Feed ─────────────────────────────────────
    ws_feed: WebSocketFeed | None = None
    if use_websocket:
        print(f"\n{CYN}[ws] Starting WebSocket live feed...{RST}")
        flog.info("Starting WebSocket feed")
        try:
            ws_feed = WebSocketFeed(client, active_stocks)
            ws_feed.start()
            if ws_feed.is_connected:
                print(f"{GRN}[ws] WebSocket connected — zero REST polling during live{RST}")
            else:
                print(f"{YLW}[ws] WebSocket connecting in background — REST fallback active{RST}")
        except Exception as ws_exc:
            print(f"{YLW}[ws] WebSocket failed ({ws_exc}) — using REST polling{RST}")
            flog.warning("WebSocket start failed: %s — REST fallback", ws_exc)
            ws_feed = None
    else:
        print(f"{DIM}[ws] WebSocket disabled — using REST polling{RST}")
        flog.info("WebSocket disabled by --no-websocket flag")

    data_mode = "WebSocket" if ws_feed else "REST"

    send_telegram(
        f"\U0001f514 *Signal Monitor Started — Prearmed PDL*\n"
        f"Capital: ₹{capital:,.0f}\n"
        f"Stocks: {syms}\n"
        f"Variants: Baseline, BodyClose50\n"
        f"Trigger: PDL ± {PDL_PREARM_BUFFER_ATR:.1f}× ATR | SL: {PDL_SL_MULT}× ATR | RR: 1:{RR}\n"
        f"Schedule: 09:30–13:00 | Data: {data_mode}"
    )

    cooldowns: dict[tuple[str, str], datetime] = {}
    open_positions: dict[tuple[str, str], dict] = {}
    signal_counts: dict[tuple[str, str, str], int] = {}
    pdl_dirs_used: dict[tuple[str, str], set[str]] = {}
    daily_sl_counts: dict[tuple[str, str], int] = {}
    last_processed_bar_ts: dict[str, pd.Timestamp | None] = {
        sym: stock_data[sym].index[-1] if sym in stock_data and not stock_data[sym].empty else None
        for sym, _ in STOCKS
    }
    today_signals: list[dict] = []
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
                last_processed_bar_ts = {
                    sym: stock_data[sym].index[-1] if sym in stock_data and not stock_data[sym].empty else None
                    for sym, _ in STOCKS
                }
                today_signals.clear()
                scan_count = 0
                last_scan_candle = -1
                last_phase = ""
                print(f"\n{CYN}[new day] {current_date}{RST}")
                print(f"{CYN}[pdl] Refreshing prev-day levels...{RST}")
                pdl_levels = fetch_prev_day_levels(stock_data, current_date)

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
                    log_file = SIGNALS_DIR / f"signals_{current_date}.csv"
                    print(f"\n{DIM}[{now:%H:%M}] {lbl}. {n} signal(s) today.{RST}")
                    flog.info("=" * 70)
                    flog.info("SHUTDOWN  %d signal(s), %d scan cycles", n, scan_count)
                    if today_signals:
                        for s in today_signals:
                            print(f"  {s['time']:%H:%M}  {s['dir']:>5}  {s['sym']:<11}  {s['variant']:<11}  @ ₹{s['price']:,.2f}")
                            flog.info(
                                "  SUMMARY  %s  %5s  %-11s  %-11s  @ %.2f  SL=%.2f  TP=%.2f",
                                s["time"].strftime("%H:%M"), s["dir"], s["sym"],
                                s["variant"], s["price"], s["sl"], s["tp"],
                            )
                        print(f"{DIM}  Log saved: {log_file}{RST}")
                    if open_positions:
                        print(f"\n  {YLW}{BLD}⚠ OPEN POSITIONS AT SHUTDOWN:{RST}")
                        op_lines = []
                        for (variant_key, op_sym), op_pos in open_positions.items():
                            op_sec = next((sec for s, sec in STOCKS if s == op_sym), "")
                            variant_label = next(
                                (cfg["label"] for cfg in SIGNAL_VARIANTS if cfg["key"] == variant_key),
                                variant_key,
                            )
                            op_qty = int(op_pos.get("qty", 0) or 0)
                            op_unreal_per_share = (
                                (stock_data[op_sym].iloc[-1]["close"] - op_pos["entry_price"])
                                if op_pos["direction"] == "LONG"
                                else (op_pos["entry_price"] - stock_data[op_sym].iloc[-1]["close"])
                            ) if op_sym in stock_data and not stock_data[op_sym].empty else 0
                            op_unreal = op_unreal_per_share * op_qty
                            pnl_s = f"+₹{op_unreal:,.2f}" if op_unreal >= 0 else f"-₹{abs(op_unreal):,.2f}"
                            print(f"  {op_sym:<14} {op_pos['direction']:>5}  {variant_label:<11}  entry ₹{op_pos['entry_price']:,.2f}  "
                                  f"SL ₹{op_pos['sl']:,.2f}  TP ₹{op_pos['tp']:,.2f}  unrealized {pnl_s}")
                            flog.warning("OPEN AT SHUTDOWN  %-11s  %-11s  %s  entry=%.2f  SL=%.2f  TP=%.2f",
                                         op_sym, variant_label, op_pos["direction"], op_pos["entry_price"],
                                         op_pos["sl"], op_pos["tp"])
                            op_lines.append(f"{op_sym} {op_pos['direction']} {variant_label} @ ₹{op_pos['entry_price']:,.2f}")
                        send_telegram(
                            f"\u26a0\ufe0f *OPEN POSITIONS AT SHUTDOWN:*\n"
                            + "\n".join(op_lines)
                            + "\n\nManually check SL/TP bracket orders."
                        )

                    flog.info("=" * 70)
                    send_telegram(f"\U0001f4f4 Shutdown. {n} signal(s) today.")
                    if ws_feed is not None:
                        ws_feed.stop()
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
            scan_bar_completed_ts = pd.Timestamp(
                _base + timedelta(seconds=_candle_idx * 180)
            )
            _wait_poll = (_poll_at - datetime.now()).total_seconds()
            if _wait_poll > 0:
                _time.sleep(_wait_poll)
            scan_delay = max(0.0, (datetime.now() - _poll_at).total_seconds())
            if scan_delay > 5:
                flog.warning("SCAN delay %.1fs beyond target %s", scan_delay, _poll_at.strftime("%H:%M:%S"))

            missed = _candle_idx - last_scan_candle - 1
            if missed > 0:
                flog.warning("MISSED %d candle(s) (idx %d→%d)", missed, last_scan_candle, _candle_idx)
            last_scan_candle = _candle_idx

            # ── Scan cycle ────────────────────────────────────────────
            scan_count += 1
            scan_start = _time.time()
            is_active = phase in ("ACTIVE_AM", "ACTIVE_PM")
            dashboard: list[dict] = []
            flog.info(
                "SCAN #%d  %s  phase=%s  active=%s  missed=%d",
                scan_count, now.strftime("%H:%M:%S"), phase, is_active, missed,
            )

            # Phase A: Data fetch
            active_syms = [s for s, _ in STOCKS if s in stock_data]
            ws_ok = ws_feed is not None and ws_feed.is_connected
            ws_stale = ws_feed is not None and ws_feed.seconds_since_last_tick > WS_STALE_SEC
            first_scan = scan_count == 1
            ws_recovered = ws_feed is not None and ws_feed.had_outage

            gap_fill_reasons: list[str] = []
            if first_scan:
                gap_fill_reasons.append("first scan")
            if ws_recovered:
                gap_fill_reasons.append("WS outage recovery")
            if missed > 0:
                gap_fill_reasons.append(f"missed {missed} candle(s)")

            need_gap_fill = bool(gap_fill_reasons)
            if need_gap_fill:
                reason = ", ".join(gap_fill_reasons)
                flog.info("FETCH  REST gap-fill (%s)", reason)
                for sym in active_syms:
                    try:
                        _rate_limit_wait()
                        _fetch_latest(client, sym, stock_data, now, retries=2)
                    except Exception as exc:
                        flog.warning("GAP-FILL  %-11s  %s", sym, exc)
                if ws_recovered and ws_feed is not None:
                    ws_feed.clear_outage()

            if ws_ok and not ws_stale:
                # ── WebSocket path: read completed bars from local buffer ──
                ws_flushed = ws_feed.flush_elapsed_bars(scan_bar_completed_ts.to_pydatetime())
                n_bars = 0
                for sym in active_syms:
                    new_bars = ws_feed.get_completed_bars(sym)
                    if not new_bars.empty:
                        combined = pd.concat([stock_data[sym], new_bars])
                        stock_data[sym] = combined[~combined.index.duplicated(keep="last")].sort_index()
                        n_bars += len(new_bars)
                flog.info(
                    "FETCH  WS  %d new bars (%d clock-flushed) in %.3fs",
                    n_bars, ws_flushed, _time.time() - scan_start,
                )
            else:
                # ── REST fallback: parallel fetch (rate-limited 3 rps) ─────
                if ws_feed is not None and ws_stale:
                    flog.warning("WS stale (%.0fs since last tick) — REST fallback",
                                 ws_feed.seconds_since_last_tick)
                fetch_ok: dict[str, bool] = {}
                need_reconnect = False

                with ThreadPoolExecutor(max_workers=3) as pool:
                    futures = {
                        pool.submit(_fetch_stock_threaded, client, sym, stock_data, now): sym
                        for sym in active_syms
                    }
                    try:
                        for future in as_completed(futures, timeout=30):
                            sym_f = futures[future]
                            try:
                                _, ok, exc = future.result()
                                fetch_ok[sym_f] = ok
                                if not ok and exc:
                                    exc_s = str(exc).lower()
                                    if "invalid" in exc_s or "session" in exc_s:
                                        need_reconnect = True
                                    elif "rate" in exc_s or "toomanyrequests" in exc_s:
                                        flog.warning("API  %-11s  rate-limited, using cache", sym_f)
                                    else:
                                        flog.warning("API  %-11s  fetch error: %s", sym_f, exc)
                            except Exception as fut_exc:
                                fetch_ok[sym_f] = False
                                flog.warning("API  %-11s  future error: %s", sym_f, fut_exc)
                    except TimeoutError:
                        for f, s in futures.items():
                            if not f.done():
                                fetch_ok[s] = False
                                flog.warning("API  %-11s  scan deadline (30s), using cache", s)

                if need_reconnect:
                    try:
                        client.connect()
                        flog.info("API session reconnected")
                        if ws_feed is not None:
                            ws_feed.stop()
                            ws_feed = WebSocketFeed(client, active_stocks)
                            ws_feed.start()
                            flog.info("WS feed restarted after session reconnect")
                    except Exception as rexc:
                        flog.error("API reconnect FAILED: %s", rexc)

                n_ok = sum(1 for v in fetch_ok.values() if v)
                flog.info("FETCH  REST  %d/%d OK in %.1fs", n_ok, len(active_syms), _time.time() - scan_start)

            # Phase B: Sequential signal processing
            for sym, sec in STOCKS:
                if sym not in stock_data:
                    continue

                df = compute_indicators(stock_data[sym])
                pdl_h, pdl_l = pdl_levels.get(sym, (None, None))
                latest_dashboard_result = None
                last_seen = last_processed_bar_ts.get(sym)
                pending_ts = list(df.index[df.index > last_seen]) if last_seen is not None else list(df.index)

                if not pending_ts:
                    latest_dashboard_result = scan_bar(
                        sym, sec, df, -1, capital, pdl_h, pdl_l,
                        pdl_dirs_used.setdefault(("baseline", sym), set()),
                        variant_label="Baseline",
                        body_close_min=None,
                        trigger_name="PDL_BASE",
                    )
                    latest_dashboard_result["signal"] = None
                else:
                    latest_pending_ts = pending_ts[-1]
                    for bar_ts in pending_ts:
                        bar_idx = df.index.get_loc(bar_ts)
                        bar_row = df.iloc[bar_idx]
                        bar_completed_ts = bar_completed_at(bar_ts)
                        is_current_scan_bar = bar_completed_ts == scan_bar_completed_ts
                        bar_phase = get_phase(bar_completed_ts.time())
                        bar_is_active = bar_phase in ("ACTIVE_AM", "ACTIVE_PM")
                        is_latest_bar = bar_ts == latest_pending_ts

                        for cfg in SIGNAL_VARIANTS:
                            state_key = (cfg["key"], sym)
                            sym_pdl_dirs = pdl_dirs_used.setdefault(state_key, set())

                            if state_key in open_positions:
                                pos = open_positions[state_key]
                                hit = _check_bracket_hit(pos, bar_row["high"], bar_row["low"])
                                if hit:
                                    flog.info(
                                        "BRACKET  %-11s  %-11s  %s hit at %s — position closed",
                                        sym, cfg["label"], hit, bar_completed_ts.strftime("%H:%M"),
                                    )
                                    if hit == "SL":
                                        daily_sl_counts[state_key] = daily_sl_counts.get(state_key, 0) + 1
                                        flog.info(
                                            "SL_COUNT  %-11s  %-11s  %d / %d",
                                            sym, cfg["label"], daily_sl_counts[state_key], MAX_SL_PER_DAY,
                                        )
                                    del open_positions[state_key]
                                else:
                                    pos["last_bracket_ts"] = bar_ts

                            result = scan_bar(
                                sym, sec, df, bar_idx, capital, pdl_h, pdl_l, sym_pdl_dirs,
                                variant_label=cfg["label"],
                                body_close_min=cfg["body_close_min"],
                                trigger_name=cfg["trigger_name"],
                            )
                            if cfg["key"] == "baseline" and is_latest_bar:
                                latest_dashboard_result = result

                            r = result
                            rsi_str = f"{r['rsi']:.0f}" if not pd.isna(r.get("rsi", float("nan"))) else "-"
                            vr = r.get("vol_r", float("nan"))
                            vol_str = f"{vr:.1f}x" if not pd.isna(vr) else "-"
                            raw_sig = r.get("signal") or "-"
                            raw_trig = r.get("trigger") or "-"

                            on_cooldown = state_key in cooldowns and bar_completed_ts < cooldowns[state_key]
                            already_in_position = state_key in open_positions
                            session_key = "AM" if bar_phase == "ACTIVE_AM" else "PM"
                            count_key = (cfg["key"], sym, session_key)
                            at_signal_cap = signal_counts.get(count_key, 0) >= MAX_SIGNALS_PER_STOCK
                            at_sl_cap = daily_sl_counts.get(state_key, 0) >= MAX_SL_PER_DAY

                            if (
                                bar_is_active
                                and result["signal"]
                                and not on_cooldown
                                and not already_in_position
                                and not at_signal_cap
                                and not at_sl_cap
                            ):
                                if not is_latest_bar or not is_current_scan_bar:
                                    cooldowns[state_key] = bar_completed_ts + COOLDOWN
                                    sym_pdl_dirs.add(r["signal"])
                                    stale_reason = (
                                        "backlog"
                                        if not is_latest_bar
                                        else f"late_bar(expected={scan_bar_completed_ts:%H:%M})"
                                    )
                                    flog.warning(
                                        "STALE SIGNAL  %-11s  %-11s  %s[%s]  bar=%s  detected=%s  %s — skipped live alert",
                                        sym, r["variant"], r["signal"], r["trigger"],
                                        bar_completed_ts.strftime("%H:%M"), now.strftime("%H:%M:%S"),
                                        stale_reason,
                                    )
                                    result["signal"] = None
                                    continue

                                print_signal(
                                    sym, sec, r["signal"], r["trigger"], r["price"],
                                    r["atr"], r["qty"], r["sl"], r["tp"], r["rsi"],
                                    r["vwap"], r["vol_ratio"], r.get("sl_mult", 1.0),
                                    variant=r["variant"],
                                )
                                send_telegram(format_tg_signal(
                                    sym, sec, r["signal"], r["trigger"], r["variant"], r["price"],
                                    r["atr"], r["qty"], r["sl"], r["tp"], r["rsi"],
                                    r["vwap"], r["vol_ratio"], r.get("sl_mult", 1.0),
                                ))
                                log_signal(
                                    sym, sec, r["signal"], r["trigger"], r["variant"], r["price"],
                                    r["sl"], r["tp"], r["atr"], r["qty"], r["rsi"],
                                    r["vwap"], r["vol_ratio"], phase=bar_phase,
                                )
                                cooldowns[state_key] = bar_completed_ts + COOLDOWN
                                signal_counts[count_key] = signal_counts.get(count_key, 0) + 1
                                sym_pdl_dirs.add(r["signal"])
                                open_positions[state_key] = {
                                    "direction": r["signal"],
                                    "entry_price": r["price"],
                                    "entry_atr": r["atr"],
                                    "qty": r["qty"],
                                    "sl": r["sl"],
                                    "tp": r["tp"],
                                    "entry_scan": scan_count,
                                    "entry_time": bar_completed_ts,
                                    "last_bracket_ts": bar_ts,
                                    "variant": r["variant"],
                                }
                                today_signals.append({
                                    "sym": sym,
                                    "dir": r["signal"],
                                    "trigger": r["trigger"],
                                    "variant": r["variant"],
                                    "price": r["price"],
                                    "sl": r["sl"],
                                    "tp": r["tp"],
                                    "time": now,
                                })
                                flog.info(
                                    "  %-11s  %-11s  C=%.2f  VWAP=%.2f  RSI=%s  Vol=%s  → ★ %s[%s]  SL=%.2f  TP=%.2f  Qty=%d",
                                    sym, r["variant"], r["close"], r["vwap"], rsi_str,
                                    vol_str, r["signal"], r["trigger"], r["sl"], r["tp"], r["qty"],
                                )
                            else:
                                reason = ""
                                if not bar_is_active:
                                    reason = "inactive_phase"
                                elif already_in_position:
                                    reason = "in_position"
                                elif on_cooldown:
                                    reason = f"cooldown_until_{cooldowns[state_key]:%H:%M}"
                                elif at_signal_cap:
                                    reason = f"max_signals_{session_key}({signal_counts.get(count_key, 0)})"
                                elif at_sl_cap:
                                    reason = f"sl_cap({daily_sl_counts.get(state_key, 0)}/{MAX_SL_PER_DAY})"
                                elif raw_sig != "-":
                                    reason = "blocked"
                                result["signal"] = None
                                flog.debug(
                                    "  %-11s  %-11s  C=%.2f  VWAP=%.2f  RSI=%s  Vol=%s  → %s[%s]  %s",
                                    sym, r["variant"], r["close"], r["vwap"], rsi_str,
                                    vol_str, raw_sig, raw_trig, reason,
                                )

                    last_processed_bar_ts[sym] = latest_pending_ts

                if latest_dashboard_result is not None:
                    dashboard.append(latest_dashboard_result)

            scan_elapsed = _time.time() - scan_start
            flog.info("SCAN #%d complete in %.1fs", scan_count, scan_elapsed)
            print_dashboard(dashboard, phase)

    except KeyboardInterrupt:
        n = len(today_signals)
        flog.info("STOPPED MANUALLY  %d signal(s), %d scans", n, scan_count)
        for s in today_signals:
            flog.info(
                "  SUMMARY  %s  %5s  %-11s  %-11s  @ %.2f  [%s]",
                s["time"].strftime("%H:%M"), s["dir"], s["sym"],
                s.get("variant", "?"), s["price"], s.get("trigger", "?"),
            )
        print(f"\n\n{YLW}[stopped] {n} signal(s) generated today:{RST}")
        for s in today_signals:
            trig = s.get("trigger", "?")
            print(f"  {s['time']:%H:%M}  {s['dir']:>5}  [{trig}]  {s['sym']:<11}  {s.get('variant', '?'):<11}  @ ₹{s['price']:,.2f}")
        if ws_feed is not None:
            ws_feed.stop()
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
    parser.add_argument(
        "--no-websocket", action="store_true",
        help="Disable WebSocket feed, use REST polling only",
    )
    args = parser.parse_args()

    print(f"\n{BLD}NSE Intraday Scalping — Live Signal Generator{RST}")
    print(f"{DIM}Prearmed PDL ({PDL_PREARM_BUFFER_ATR:.1f}×ATR trigger buffer, SL {PDL_SL_MULT}×ATR)")
    print(f"Variants: Baseline + BodyClose50 | RR 1:{RR} | Risk {RISK_PCT * 100:.1f}% | Max {MAX_SL_PER_DAY} SL/stock/day{RST}\n")

    if args.dry_run:
        dry_run(args.capital)
    else:
        try:
            live(args.capital, use_websocket=not args.no_websocket)
        except KeyboardInterrupt:
            flog.info("SESSION END  user interrupt")
            print(f"\n{YLW}Session ended by user.{RST}")
        except Exception as exc:
            flog.error("SESSION CRASH  %s: %s", type(exc).__name__, exc)
            print(f"\n{RED}CRASH: {exc}{RST}")
            raise


if __name__ == "__main__":
    main()

"""
Live Signal Generator for NSE Intraday Scalping Strategy — PREARMED PDL

Streams live data via Angel One WebSocket, computes indicators in
real-time (pandas/numpy), and sends BUY/SHORT alerts to console + Telegram.

STRATEGY:
  PDL Breakout — trigger is placed just beyond yesterday's High/Low.
  The trigger buffer (0.1× ATR) uses the current bar's ATR, so it
  adjusts slightly through the day as volatility evolves.
  Signal fires when the completed 3-minute bar trades through that trigger
  and still closes through the base prev-day level.

Daily Lifecycle:
  Phase 1  09:14  Warm-Up   — connect, fetch 5 days of 3-min data
  Phase 2  09:15  Silent    — track indicators every 3 min, NO signals
  Phase 3  09:30  Active    — trading window, signals enabled
  Phase 4  13:00  Shutdown  — stop signals, track only until close

Entry   : Prearmed PDL trigger = prev-day High/Low ± 0.1× ATR
Confirm : Signal bar must close beyond trigger by 0.05× ATR
          LONG after 10:00 must close beyond trigger by 0.20× ATR
          SHORT after 10:00 must close beyond trigger by 0.10× ATR
Exit    : SL = 1.0× ATR, TP = 1.5R, max 2 signals per stock per session
Sizing  : 1.5% risk per trade, 5× leverage cap
Guard   : Max 2 SL hits per stock per day — blocks further entries
Fill    : Min 50% fill ratio AND ₹100 gross TP to keep partial fills

Usage:
    python live_signals.py                  # default ₹25,000 capital
    python live_signals.py --capital 100000 # custom capital
    python live_signals.py --dry-run        # scan last session, no live loop
"""

import argparse
import csv
import ipaddress
import json
import logging
import os
import sys
import threading
import time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from event_calendar import get_excluded_stocks
from fetch_data import AngelOneClient, load_csv
from order_manager import (
    place_bracket_order, wait_for_fill_or_cancel,
    place_sl_order, place_tp_order, cancel_order, place_market_exit_order,
    round_to_tick, _entry_buffer,
)
from websocket_feed import WebSocketFeed

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))


# ─── Stock Configuration ──────────────────────────────────────────────────

STOCKS = [
    ("TCS", "IT"),
    ("HCLTECH", "IT"),
    ("SHRIRAMFIN", "Finance"),
    ("NTPC", "Power"),
    ("BAJAJ-AUTO", "Auto"),
    ("ADANIPORTS", "Ports"),
    ("HDFCLIFE", "Insurance"),
    ("COALINDIA", "Mining"),
    ("ADANIENT", "Infra"),
    ("MARUTI", "Auto"),
]

# ─── Strategy Parameters ──────────────────────────────────────────────────

ATR_PERIOD = 14

RR = 1.5
RISK_PCT = 0.015
LEV_CAP = 5.0

PDL_PREARM_BUFFER_ATR = 0.10
PDL_CLOSE_EXT_ATR = 0.05
POST_10_LONG_CLOSE_EXT_ATR = 0.20
POST_10_SHORT_CLOSE_EXT_ATR = 0.10
PDL_SL_MULT = 1.0
MAX_SL_PER_DAY = 2

SIGNAL_VARIANTS = [
    {"key": "baseline", "label": "Baseline", "body_close_min": None, "trigger_name": "PDL_BASE"},
]


# ─── Time Rules & Phases ──────────────────────────────────────────────────

MKT_OPEN = time(9, 15)
## Update market close time here
ENTRY_AM = (time(9, 30), time(13, 00))
ENTRY_PM = None
MKT_CLOSE = time(15, 0)

BAR_INTERVAL = timedelta(minutes=3)
COOLDOWN = timedelta(minutes=30)
MAX_SIGNALS_PER_STOCK = 2
WARMUP_3MIN_DAYS = 5
POLL_BUFFER_SEC = 1
WS_STALE_SEC = 20
EARLY_STRONG_CLOSE_END = time(10, 0)
EARLY_BODY_CLOSE_MIN = 0.60
POST_10_LONG_CLOSE_EXT_START = time(10, 0)
POST_10_SHORT_CLOSE_EXT_START = time(10, 0)

DATA_DIR = Path(__file__).parent / "data"
LOG_DIR = Path(__file__).parent / "logs"
LIVE_LOG_DIR = LOG_DIR / "live_signals"
SIGNALS_DIR = LOG_DIR / "signals"
STATE_FILE = LOG_DIR / ".live_state.json"

MAX_CONCURRENT_POSITIONS = 3
MAX_DAILY_LOSS = 3000.0
ENABLE_AUTO_TRADE = True
PARTIAL_EXIT_TIMEOUT_SEC = 15
MIN_FILL_RATIO = 0.50
MIN_ACTUAL_GROSS_TARGET = 100.0
MIN_REMAINING_REWARD_ATR = 0.3

IST = ZoneInfo("Asia/Kolkata")


def _now() -> datetime:
    """Return current time in IST as a naive datetime (for consistency with pandas)."""
    return datetime.now(IST).replace(tzinfo=None)

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
        LIVE_LOG_DIR / f"live_{_now():%Y-%m-%d}.log", encoding="utf-8",
    )
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
    flog.addHandler(fh)


flog = _init_file_logger()


def _save_state(session_date, cooldowns, pdl_dirs_used, daily_sl_counts,
                open_positions, pending_entries, signal_counts, today_signals,
                realized_pnl: float = 0.0):
    """Persist intraday state to disk for restart recovery."""
    try:
        serialized_positions = {}
        for (vk, sym), pos in open_positions.items():
            pos_copy = {}
            for pk, pv in pos.items():
                if isinstance(pv, (pd.Timestamp, datetime)):
                    pos_copy[pk] = pv.isoformat()
                else:
                    pos_copy[pk] = pv
            serialized_positions[f"{vk}|{sym}"] = pos_copy

        serialized_pending = {}
        for (vk, sym), pending in pending_entries.items():
            pending_copy = {}
            for pk, pv in pending.items():
                if isinstance(pv, (pd.Timestamp, datetime)):
                    pending_copy[pk] = pv.isoformat()
                else:
                    pending_copy[pk] = pv
            serialized_pending[f"{vk}|{sym}"] = pending_copy

        state = {
            "date": str(session_date),
            "cooldowns": {f"{k[0]}|{k[1]}": v.isoformat() for k, v in cooldowns.items()},
            "pdl_dirs_used": {f"{k[0]}|{k[1]}": list(v) for k, v in pdl_dirs_used.items()},
            "daily_sl_counts": {f"{k[0]}|{k[1]}": v for k, v in daily_sl_counts.items()},
            "signal_counts": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in signal_counts.items()},
            "open_positions": serialized_positions,
            "pending_entries": serialized_pending,
            "realized_pnl": realized_pnl,
            "n_signals": len(today_signals),
        }
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(state, indent=2))
    except Exception:
        pass


def _load_state(session_date):
    """Load persisted state if it exists and matches the session date."""
    if not STATE_FILE.exists():
        return None
    try:
        state = json.loads(STATE_FILE.read_text())
        if state.get("date") != str(session_date):
            return None
        cooldowns = {}
        for k, v in state.get("cooldowns", {}).items():
            parts = k.split("|", 1)
            cooldowns[(parts[0], parts[1])] = datetime.fromisoformat(v)
        pdl_dirs = {}
        for k, v in state.get("pdl_dirs_used", {}).items():
            parts = k.split("|", 1)
            pdl_dirs[(parts[0], parts[1])] = set(v)
        sl_counts = {}
        for k, v in state.get("daily_sl_counts", {}).items():
            parts = k.split("|", 1)
            sl_counts[(parts[0], parts[1])] = v
        sig_counts = {}
        for k, v in state.get("signal_counts", {}).items():
            parts = k.split("|", 2)
            sig_counts[(parts[0], parts[1], parts[2])] = v

        positions = {}
        for k, pos in state.get("open_positions", {}).items():
            parts = k.split("|", 1)
            for time_key in ("entry_time", "last_bracket_ts"):
                if time_key in pos and isinstance(pos[time_key], str):
                    try:
                        pos[time_key] = pd.Timestamp(pos[time_key])
                    except Exception:
                        pass
            positions[(parts[0], parts[1])] = pos

        pending_entries = {}
        for k, pending in state.get("pending_entries", {}).items():
            parts = k.split("|", 1)
            for time_key in ("entry_time", "bar_ts"):
                if time_key in pending and isinstance(pending[time_key], str):
                    try:
                        pending[time_key] = pd.Timestamp(pending[time_key])
                    except Exception:
                        pass
            pending_entries[(parts[0], parts[1])] = pending

        return {
            "cooldowns": cooldowns,
            "pdl_dirs_used": pdl_dirs,
            "daily_sl_counts": sl_counts,
            "signal_counts": sig_counts,
            "open_positions": positions,
            "pending_entries": pending_entries,
            "realized_pnl": float(state.get("realized_pnl", 0.0)),
        }
    except Exception:
        return None


_LOG_COLUMNS = [
    "date", "time", "symbol", "sector", "direction", "trigger", "variant", "price",
    "sl", "tp", "atr", "qty", "risk", "phase",
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
    if t < ENTRY_AM[1]:
        return "ACTIVE_AM"
    bar_grace = (datetime.combine(datetime.min, ENTRY_AM[1]) + BAR_INTERVAL).time()
    if t <= bar_grace:
        return "ACTIVE_AM"
    if ENTRY_PM is None:
        return "SHUTDOWN"
    if t < ENTRY_PM[0]:
        return "MIDDAY"
    if t < ENTRY_PM[1]:
        return "ACTIVE_PM"
    bar_grace_pm = (datetime.combine(datetime.min, ENTRY_PM[1]) + BAR_INTERVAL).time()
    if t <= bar_grace_pm:
        return "ACTIVE_PM"
    return "SHUTDOWN"


# ─── Terminal Colors ───────────────────────────────────────────────────────

GRN, RED, YLW, CYN = "\033[92m", "\033[91m", "\033[93m", "\033[96m"
BLD, DIM, RST = "\033[1m", "\033[2m", "\033[0m"


# ─── Telegram ─────────────────────────────────────────────────────────────

TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")
ANGEL_REGISTERED_IPS = [
    ip.strip() for ip in os.getenv("ANGEL_REGISTERED_IPS", "").split(",") if ip.strip()
]

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


def _fetch_public_ip() -> str | None:
    """Best-effort public IP lookup for startup diagnostics."""
    for url in ("https://api.ipify.org", "https://checkip.amazonaws.com"):
        try:
            resp = requests.get(url, timeout=5)
            if not resp.ok:
                continue
            candidate = resp.text.strip()
            ipaddress.ip_address(candidate)
            return candidate
        except Exception:
            continue
    return None


def _startup_order_path_check(auto_trade_active: bool) -> None:
    """Print current public IP and warn if it is not in the registered allowlist."""
    public_ip = _fetch_public_ip()
    if public_ip:
        print(f"{CYN}[net] Public IP: {public_ip}{RST}")
        flog.info("NET  public_ip=%s", public_ip)
    else:
        print(f"{YLW}[net] Could not determine public IP{RST}")
        flog.warning("NET  public_ip lookup failed")

    if not auto_trade_active:
        return

    if not ANGEL_REGISTERED_IPS:
        print(f"{YLW}[net] ANGEL_REGISTERED_IPS not set — cannot pre-verify IP whitelist for order placement{RST}")
        flog.warning("NET  ANGEL_REGISTERED_IPS not configured; cannot pre-verify order IP")
        return

    registered = ", ".join(ANGEL_REGISTERED_IPS)
    if public_ip and public_ip in ANGEL_REGISTERED_IPS:
        print(f"{GRN}[net] Public IP matches registered Angel IPs ({registered}){RST}")
        flog.info("NET  public_ip matched registered_ips=%s", registered)
    else:
        print(f"{RED}{BLD}[net] WARNING — current IP is not in ANGEL_REGISTERED_IPS{RST}")
        print(f"{RED}      Registered: {registered}{RST}")
        print(f"{RED}      Orders are likely to fail with 'Unregistered IP address'.{RST}")
        flog.error(
            "NET  public_ip mismatch current=%s registered=%s — order placement likely to fail",
            public_ip, registered,
        )
        send_telegram(
            f"⚠️ *Startup IP Warning*\n"
            f"Current public IP: `{public_ip or 'unknown'}`\n"
            f"Registered IPs: `{registered}`\n"
            f"Auto-trade orders are likely to fail with *Unregistered IP address*."
        )


# ─── Signal Log ──────────────────────────────────────────────────────────


def log_signal(
    sym, sec, direction, trigger, variant, price, sl, tp, atr, qty,
    ts=None, phase="",
) -> None:
    """Append one signal row to the daily CSV log file."""
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = ts or _now()
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


def _update_signal_csv_row(
    *,
    signal_time: datetime | pd.Timestamp | None,
    sym: str,
    direction: str,
    variant: str,
    price: float | None = None,
    sl: float | None = None,
    tp: float | None = None,
) -> None:
    """Best-effort in-place update of an existing daily signal row.

    We keep the CSV schema stable and only refresh price/sl/tp for the matching
    row so post-session review reflects broker-confirmed reconciliations.
    """
    if signal_time is None:
        return
    stamp = pd.Timestamp(signal_time)
    log_file = SIGNALS_DIR / f"signals_{stamp:%Y-%m-%d}.csv"
    if not log_file.exists():
        return
    try:
        with open(log_file, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return
        target_time = stamp.strftime("%H:%M:%S")
        updated = False
        for row in rows:
            if (
                row.get("time") == target_time
                and row.get("symbol") == sym
                and row.get("direction") == direction
                and row.get("variant", "") == (variant or "")
            ):
                if price is not None:
                    row["price"] = f"{float(price):.2f}"
                if sl is not None:
                    row["sl"] = f"{float(sl):.2f}"
                if tp is not None:
                    row["tp"] = f"{float(tp):.2f}"
                updated = True
                break
        if updated:
            with open(log_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=_LOG_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
    except Exception as exc:
        flog.warning("SIGNAL CSV UPDATE FAILED  %s %s %s: %s", sym, direction, target_time, exc)


def _trigger_label(trigger: str) -> str:
    return {
        "PDL_BASE": "Prearmed PDL",
    }.get(trigger, trigger)


# ─── Indicator Engine ─────────────────────────────────────────────────────


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate ATR on a 3-min OHLCV DataFrame (the only indicator used)."""
    out = df.copy()
    pc = out["close"].shift(1)
    tr = np.maximum(out["high"], pc) - np.minimum(out["low"], pc)
    out["atr"] = tr.ewm(
        alpha=1.0 / ATR_PERIOD, min_periods=ATR_PERIOD, adjust=False
    ).mean()
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
    bar_completed_time: time | None = None,
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
    tick_tol = atr * 0.05

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
        long_close_extension = atr * _effective_close_extension_atr("LONG", bar_completed_time)
        if ("LONG" not in dirs
                and prev_close <= prev_day_high + tick_tol
                and h >= long_trigger
                and c >= long_trigger + long_close_extension
                and body_close_ok("LONG")):
            return "LONG", trigger_name
    if prev_day_low is not None:
        short_trigger = prev_day_low - buffer
        short_close_extension = atr * _effective_close_extension_atr("SHORT", bar_completed_time)
        if ("SHORT" not in dirs
                and prev_close >= prev_day_low - tick_tol
                and l <= short_trigger
                and c <= short_trigger - short_close_extension
                and body_close_ok("SHORT")):
            return "SHORT", trigger_name

    return None, None


def calc_qty(capital: float, atr: float, price: float) -> int:
    risk_q = int(capital * RISK_PCT / atr)
    max_q = int(capital * LEV_CAP / price)
    return max(min(risk_q, max_q), 1)


def gross_target_rupees(direction: str, entry_price: float, tp_price: float, qty: int) -> float:
    """Return the gross TP rupee potential for a filled position."""
    if direction == "LONG":
        gross = (tp_price - entry_price) * qty
    else:
        gross = (entry_price - tp_price) * qty
    return max(float(gross), 0.0)


def _effective_body_close_min(
    base_body_close_min: float | None,
    bar_completed_time: time,
) -> float | None:
    """Return the body-close requirement for a completed bar.

    Baseline behavior now requires a stronger close only during the first 30
    minutes of the active session (09:30–10:00). Variant-specific filters, if
    supplied, take precedence.
    """
    if base_body_close_min is not None:
        return base_body_close_min
    if ENTRY_AM:
        early_cutoff = min(ENTRY_AM[1], EARLY_STRONG_CLOSE_END)
        if ENTRY_AM[0] <= bar_completed_time <= early_cutoff:
            return EARLY_BODY_CLOSE_MIN
    return None


def _effective_close_extension_atr(
    direction: str,
    bar_completed_time: time | None,
) -> float:
    """Return the close-beyond-trigger extension in ATR units for a bar."""
    if (
        direction == "LONG"
        and bar_completed_time is not None
        and bar_completed_time >= POST_10_LONG_CLOSE_EXT_START
    ):
        return POST_10_LONG_CLOSE_EXT_ATR
    if (
        direction == "SHORT"
        and bar_completed_time is not None
        and bar_completed_time >= POST_10_SHORT_CLOSE_EXT_START
    ):
        return POST_10_SHORT_CLOSE_EXT_ATR
    return PDL_CLOSE_EXT_ATR


# ─── Display ──────────────────────────────────────────────────────────────


def print_signal(
    sym, sec, direction, trigger, price, atr, qty, sl, tp,
    sl_mult, ts=None, variant="",
):
    clr = GRN if direction == "LONG" else RED
    arrow = "▲" if direction == "LONG" else "▼"
    trig_lbl = _trigger_label(trigger)
    stamp = (ts or _now()).strftime("%H:%M:%S")
    print(f"\n{'=' * 60}")
    print(f"{clr}{BLD}  {arrow} {direction} [{trig_lbl}] — {sym} ({sec}) · {variant}  [{stamp}]{RST}")
    print(f"{'=' * 60}")
    print(f"  Price  : ₹{price:,.2f}")
    print(f"  SL     : ₹{sl:,.2f}  ({sl_mult}× ATR = ₹{atr * sl_mult:.2f})")
    print(f"  TP     : ₹{tp:,.2f}  (RR 1:{RR})")
    print(f"  Qty    : {qty}  (risk ₹{qty * atr * sl_mult:,.0f})")
    print(f"{'=' * 60}\n")


def format_tg_signal(sym, sec, direction, trigger, variant, price, atr, qty, sl, tp, sl_mult):
    icon = "\U0001f7e2" if direction == "LONG" else "\U0001f534"
    trig_lbl = _trigger_label(trigger)
    return (
        f"{icon} *{direction} [{trig_lbl}] — {sym}* ({sec})\n"
        f"Price: ₹{price:,.2f}\n"
        f"SL: ₹{sl:,.2f} | TP: ₹{tp:,.2f}\n"
        f"Qty: {qty} | Risk: ₹{qty * atr * sl_mult:,.0f}\n"
        f"RR: 1:{RR}"
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


def _update_today_signal_for_order(
    today_signals: list[dict],
    *,
    order_id: str | None,
    sym: str,
    direction: str,
    variant: str,
    price: float,
    sl: float,
    tp: float,
    status: str,
    trigger: str = "PDL_BASE",
    fill_time: datetime | None = None,
    fallback_time: datetime | pd.Timestamp | None = None,
) -> None:
    """Update an existing today_signals row for an entry order, or append one."""
    if fill_time is None:
        fill_time = _now()
    for existing in reversed(today_signals):
        if order_id and existing.get("order_id") == order_id:
            existing["price"] = price
            existing["sl"] = sl
            existing["tp"] = tp
            existing["status"] = status
            existing["fill_time"] = fill_time
            existing["trigger"] = existing.get("trigger") or trigger
            return

    today_signals.append({
        "sym": sym,
        "dir": direction,
        "trigger": trigger,
        "variant": variant,
        "price": price,
        "sl": sl,
        "tp": tp,
        "time": fallback_time if fallback_time is not None else fill_time,
        "status": status,
        "order_id": order_id,
        "fill_time": fill_time,
    })


def _reconcile_late_filled_entry(
    conn,
    sym: str,
    pending: dict,
    *,
    fill_qty: int,
    fill_price: float,
) -> dict:
    """Attach protection (or flatten) for an entry that filled after timeout.

    Returns:
      status: open | closed | manual
      position: open-position dict when protection is attached
      pnl_delta: realized P&L booked during flatten/immediate broker exit
      sl_hit: True when the reconciled close was an SL
    """
    direction = pending.get("direction")
    variant = pending.get("variant", "Baseline")
    requested_qty = int(pending.get("requested_qty", 0) or 0)
    tracked_qty = int(fill_qty or requested_qty or 0)
    tracked_entry_price = float(fill_price or pending.get("signal_price") or 0.0)
    sl_price = float(pending.get("sl", 0) or 0.0)
    tp_price = float(pending.get("tp", 0) or 0.0)
    atr = float(pending.get("atr", 0) or 0.0)
    entry_oid = pending.get("entry_order_id")

    result = {"status": "manual", "position": None, "pnl_delta": 0.0, "sl_hit": False}

    if not direction or tracked_qty <= 0 or tracked_entry_price <= 0 or sl_price <= 0 or tp_price <= 0:
        flog.error(
            "LATE FILL INVALID  %-11s  direction=%s qty=%s entry=%.2f sl=%.2f tp=%.2f",
            sym, direction, tracked_qty, tracked_entry_price, sl_price, tp_price,
        )
        send_telegram(
            f"\u26a0\ufe0f *MANUAL ACTION REQUIRED* — {sym}\n"
            f"Late fill detected but reconciliation data is incomplete.\n"
            f"Entry order: `{entry_oid or 'unknown'}`\n"
            f"Check broker immediately."
        )
        return result

    fill_ratio = tracked_qty / max(requested_qty or tracked_qty, 1)
    actual_gross_target = gross_target_rupees(direction, tracked_entry_price, tp_price, tracked_qty)
    fill_is_partial = requested_qty > 0 and tracked_qty < requested_qty
    reject_late_fill = (
        fill_ratio < MIN_FILL_RATIO
        or actual_gross_target < MIN_ACTUAL_GROSS_TARGET
    )

    if reject_late_fill:
        reject_reasons = []
        if fill_ratio < MIN_FILL_RATIO:
            reject_reasons.append(f"fill_ratio={fill_ratio:.0%}<min={MIN_FILL_RATIO:.0%}")
        if actual_gross_target < MIN_ACTUAL_GROSS_TARGET:
            reject_reasons.append(f"gross_tp=₹{actual_gross_target:,.2f}<min=₹{MIN_ACTUAL_GROSS_TARGET:,.2f}")
        reject_msg = ", ".join(reject_reasons)
        flog.warning(
            "LATE FILL REJECTED  %-11s  qty=%d/%d  entry=%.2f  %s",
            sym, tracked_qty, requested_qty, tracked_entry_price, reject_msg,
        )
        exit_result = place_market_exit_order(
            conn, sym, direction, tracked_qty,
            ref_price=tracked_entry_price,
        )
        if exit_result["success"]:
            flat_result = wait_for_fill_or_cancel(
                conn,
                sym,
                exit_result["order_id"],
                timeout_sec=PARTIAL_EXIT_TIMEOUT_SEC,
            )
            exit_filled_qty = int(flat_result.get("filled_qty", 0) or 0)
            exit_fill_px = float(flat_result.get("fill_price", 0) or 0)
            if flat_result["filled"] and exit_filled_qty > 0:
                flattened_qty = min(exit_filled_qty, tracked_qty)
                if direction == "LONG":
                    flat_pnl = (exit_fill_px - tracked_entry_price) * flattened_qty
                else:
                    flat_pnl = (tracked_entry_price - exit_fill_px) * flattened_qty
                result["pnl_delta"] += flat_pnl
                if flattened_qty >= tracked_qty:
                    result["status"] = "closed"
                    send_telegram(
                        f"\u26a0\ufe0f *LATE ENTRY FILLED — FLATTENED* — {sym} {direction}\n"
                        f"Entry: `{entry_oid}` @ ₹{tracked_entry_price:,.2f} × {tracked_qty}\n"
                        f"Detected after timeout/cancel path.\n"
                        f"Reason: `{reject_msg}`\n"
                        f"Flattened: `{exit_result['order_id']}` @ ₹{exit_fill_px:,.2f}\n"
                        f"Realized: `₹{flat_pnl:+,.2f}`"
                    )
                    return result
                tracked_qty -= flattened_qty
                flog.warning(
                    "LATE FILL PARTIAL FLATTEN  %-11s  exited=%d  remaining=%d",
                    sym, flattened_qty, tracked_qty,
                )
                send_telegram(
                    f"\u26a0\ufe0f *LATE ENTRY FILLED — PARTIAL FLATTEN* — {sym} {direction}\n"
                    f"Entry: `{entry_oid}` @ ₹{tracked_entry_price:,.2f}\n"
                    f"Reason: `{reject_msg}`\n"
                    f"Flattened: `{flattened_qty}` | Remaining to protect: `{tracked_qty}`"
                )
                actual_gross_target = gross_target_rupees(direction, tracked_entry_price, tp_price, tracked_qty)
            else:
                send_telegram(
                    f"\u26a0\ufe0f *MANUAL ACTION REQUIRED* — {sym} {direction}\n"
                    f"Late fill detected after timeout/cancel path.\n"
                    f"Flatten attempt status: `{flat_result.get('status', 'unknown')}`\n"
                    f"Check broker immediately."
                )
                return result
        else:
            send_telegram(
                f"\u26a0\ufe0f *MANUAL ACTION REQUIRED* — {sym} {direction}\n"
                f"Late fill detected after timeout/cancel path.\n"
                f"Emergency flatten failed: `{exit_result['message']}`\n"
                f"Check broker immediately."
            )
            return result

    low_reward_partial_flatten = False

    # Re-check remaining reward using the actual fill price, not a synthetic
    # re-buffered entry. If full flatten fails, we fall through and protect the
    # remaining qty rather than pretending the trade is fully closed.
    if atr > 0:
        if direction == "LONG":
            remaining_reward = tp_price - tracked_entry_price
        else:
            remaining_reward = tracked_entry_price - tp_price
        min_reward = atr * MIN_REMAINING_REWARD_ATR
        if remaining_reward < min_reward:
            flog.warning(
                "LATE FILL LOW REWARD  %-11s  remaining=%.2f < min=%.2f (%.0f%% ATR) — flattening instead of protecting",
                sym, remaining_reward, min_reward, MIN_REMAINING_REWARD_ATR * 100,
            )
            exit_result = place_market_exit_order(
                conn, sym, direction, tracked_qty,
                ref_price=tracked_entry_price,
            )
            if exit_result["success"]:
                flat_result = wait_for_fill_or_cancel(
                    conn, sym, exit_result["order_id"],
                    timeout_sec=PARTIAL_EXIT_TIMEOUT_SEC,
                )
                exit_filled_qty = int(flat_result.get("filled_qty", 0) or 0)
                exit_fill_px = float(flat_result.get("fill_price", 0) or 0)
                if flat_result["filled"] and exit_filled_qty > 0:
                    flattened_qty = min(exit_filled_qty, tracked_qty)
                    pnl = (
                        (exit_fill_px - tracked_entry_price) * flattened_qty
                        if direction == "LONG"
                        else (tracked_entry_price - exit_fill_px) * flattened_qty
                    )
                    result["pnl_delta"] += pnl
                    if flattened_qty >= tracked_qty:
                        result["status"] = "closed"
                        send_telegram(
                            f"\u26a0\ufe0f *LATE ENTRY FILLED — LOW REWARD FLATTEN* — {sym} {direction}\n"
                            f"Entry @ ₹{tracked_entry_price:,.2f} × {tracked_qty}\n"
                            f"Remaining reward ₹{remaining_reward:.2f} < min ₹{min_reward:.2f}\n"
                            f"Flattened: `{exit_result['order_id']}` @ ₹{exit_fill_px:,.2f}\n"
                            f"Realized: `₹{pnl:+,.2f}`"
                        )
                        return result
                    tracked_qty -= flattened_qty
                    low_reward_partial_flatten = True
                    flog.warning(
                        "LATE FILL LOW REWARD PARTIAL FLATTEN  %-11s  exited=%d  remaining=%d",
                        sym, flattened_qty, tracked_qty,
                    )
                    send_telegram(
                        f"\u26a0\ufe0f *LATE ENTRY FILLED — LOW REWARD PARTIAL FLATTEN* — {sym} {direction}\n"
                        f"Entry @ ₹{tracked_entry_price:,.2f}\n"
                        f"Remaining reward ₹{remaining_reward:.2f} < min ₹{min_reward:.2f}\n"
                        f"Flattened: `{flattened_qty}` | Remaining protected qty: `{tracked_qty}`"
                    )
            if not low_reward_partial_flatten:
                # Only bail out on a true flatten failure (flatten order not filled
                # or flatten order placement failed). If we did a partial flatten,
                # low_reward_partial_flatten is True and we fall through to
                # place_sl_order() to protect the remaining qty.
                send_telegram(
                    f"\u26a0\ufe0f *MANUAL ACTION REQUIRED* — {sym} {direction}\n"
                    f"Late fill detected. Remaining reward too small and flatten failed.\n"
                    f"Check broker immediately."
                )
                return result

    sl_result = place_sl_order(conn, sym, direction, sl_price, tracked_qty)
    tp_result = place_tp_order(conn, sym, direction, tp_price, tracked_qty)
    if sl_result["success"] and tp_result["success"]:
        sl_oid = sl_result.get("order_id")
        tp_oid = tp_result.get("order_id")
        try:
            _imm_book = conn.orderBook()
            _imm_orders = (
                {o["orderid"]: o for o in _imm_book["data"]}
                if _imm_book and _imm_book.get("data")
                else {}
            )
            sl_filled = (
                sl_oid and sl_oid in _imm_orders
                and _imm_orders[sl_oid].get("orderstatus", "").lower() in ("complete", "filled")
            )
            tp_filled = (
                tp_oid and tp_oid in _imm_orders
                and _imm_orders[tp_oid].get("orderstatus", "").lower() in ("complete", "filled")
            )
            if tp_filled and sl_oid:
                cancel_order(conn, sl_oid, "STOPLOSS")
                pnl = (
                    (tp_price - tracked_entry_price) * tracked_qty
                    if direction == "LONG"
                    else (tracked_entry_price - tp_price) * tracked_qty
                )
                result["status"] = "closed"
                result["pnl_delta"] += pnl
                send_telegram(
                    f"\u2705 *LATE ENTRY FILLED — TP ALREADY HIT* — {sym} {direction}\n"
                    f"Entry: `{entry_oid}` @ ₹{tracked_entry_price:,.2f} × {tracked_qty}\n"
                    f"TP: `₹{tp_price:,.2f}`\n"
                    f"Realized: `₹{pnl:+,.2f}`"
                )
                return result
            if sl_filled and tp_oid:
                cancel_order(conn, tp_oid, "NORMAL")
                pnl = (
                    (sl_price - tracked_entry_price) * tracked_qty
                    if direction == "LONG"
                    else (tracked_entry_price - sl_price) * tracked_qty
                )
                result["status"] = "closed"
                result["pnl_delta"] += pnl
                result["sl_hit"] = True
                send_telegram(
                    f"\u26a0\ufe0f *LATE ENTRY FILLED — SL ALREADY HIT* — {sym} {direction}\n"
                    f"Entry: `{entry_oid}` @ ₹{tracked_entry_price:,.2f} × {tracked_qty}\n"
                    f"SL: `₹{sl_price:,.2f}`\n"
                    f"Realized: `₹{pnl:+,.2f}`"
                )
                return result
        except Exception as exc:
            flog.warning("LATE FILL IMM RECONCILE  %-11s  %s", sym, exc)

        entry_time = pending.get("entry_time")
        if isinstance(entry_time, str):
            entry_time = pd.Timestamp(entry_time)
        bar_ts = pending.get("bar_ts", entry_time)
        if isinstance(bar_ts, str):
            bar_ts = pd.Timestamp(bar_ts)

        result["status"] = "open"
        result["position"] = {
            "direction": direction,
            "entry_price": tracked_entry_price,
            "entry_atr": atr,
            "qty": tracked_qty,
            "sl": sl_price,
            "tp": tp_price,
            "entry_scan": pending.get("entry_scan", 0),
            "entry_time": entry_time if entry_time is not None else _now(),
            "last_bracket_ts": bar_ts if bar_ts is not None else _now(),
            "variant": variant,
            "sl_order_id": sl_oid,
            "tp_order_id": tp_oid,
        }
        fill_label = "PARTIAL" if fill_is_partial else "FULL"
        send_telegram(
            f"\u2705 *LATE ENTRY FILLED — PROTECTED* — {sym} {direction}\n"
            f"Fill: `{fill_label}` | Entry: `{entry_oid}` @ ₹{tracked_entry_price:,.2f} × {tracked_qty}\n"
            f"{'Full flatten did not complete; remaining qty protected.\\n' if low_reward_partial_flatten else 'Detected after timeout/cancel path.\\n'}"
            f"SL: `₹{sl_price:,.2f}` (`{sl_oid}`)\n"
            f"TP: `₹{tp_price:,.2f}` (`{tp_oid}`)"
        )
        flog.info(
            "LATE FILL PROTECTED  %-11s  %s  entry=%.2f  qty=%d  sl_oid=%s  tp_oid=%s",
            sym, direction, tracked_entry_price, tracked_qty, sl_oid, tp_oid,
        )
        return result

    flog.error(
        "LATE FILL PROTECTION FAILED  %-11s  sl=%s  tp=%s",
        sym, sl_result["success"], tp_result["success"],
    )
    emergency_exit = place_market_exit_order(
        conn, sym, direction, tracked_qty,
        ref_price=tracked_entry_price,
    )
    if emergency_exit["success"]:
        flat_result = wait_for_fill_or_cancel(
            conn,
            sym,
            emergency_exit["order_id"],
            timeout_sec=PARTIAL_EXIT_TIMEOUT_SEC,
        )
        exit_filled_qty = int(flat_result.get("filled_qty", 0) or 0)
        exit_fill_px = float(flat_result.get("fill_price", 0) or 0)
        if flat_result["filled"] and exit_filled_qty > 0:
            flattened_qty = min(exit_filled_qty, tracked_qty)
            pnl = (
                (exit_fill_px - tracked_entry_price) * flattened_qty
                if direction == "LONG"
                else (tracked_entry_price - exit_fill_px) * flattened_qty
            )
            result["pnl_delta"] += pnl
            if flattened_qty >= tracked_qty:
                result["status"] = "closed"
                send_telegram(
                    f"\u26a0\ufe0f *LATE ENTRY FILLED — FLATTENED* — {sym} {direction}\n"
                    f"Protection placement failed after late fill.\n"
                    f"Flattened: `{emergency_exit['order_id']}` @ ₹{exit_fill_px:,.2f}\n"
                    f"Realized: `₹{pnl:+,.2f}`"
                )
                return result

    send_telegram(
        f"\u26a0\ufe0f *MANUAL ACTION REQUIRED* — {sym} {direction}\n"
        f"Late fill detected after timeout/cancel path.\n"
        f"Protection placement failed and automatic flatten could not be confirmed.\n"
        f"Check broker immediately."
    )
    return result


def print_dashboard(rows: list[dict], phase: str) -> None:
    now = _now()
    phase_lbl = PHASES.get(phase, phase)
    if phase in ("ACTIVE_AM", "ACTIVE_PM"):
        phase_str = f"{GRN}{phase_lbl}{RST}"
    elif phase in ("SILENT", "MIDDAY"):
        phase_str = f"{YLW}{phase_lbl}{RST}"
    else:
        phase_str = f"{DIM}{phase_lbl}{RST}"

    print(f"\n{CYN}[{now:%H:%M:%S}]{RST} {phase_str}")
    print(f"  {'Symbol':<11} {'Close':>9}  Signal")
    print(f"  {'─' * 35}")
    for r in rows:
        sig = ""
        if r.get("signal"):
            c = GRN if r["signal"] == "LONG" else RED
            trig = r.get("trigger", "")
            sig = f"{c}{BLD}{r['signal']}[{trig}]{RST}"
        print(f"  {r['sym']:<11} ₹{r['close']:>8,.2f}  {sig}")


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
    today = _now().date()
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
                        flog.warning(
                            "WARMUP  %-11s  attempt %d failed: %s — retrying in %ds",
                            sym, attempt + 1, exc, 3 * (attempt + 1),
                        )
                        _time.sleep(3.0 * (attempt + 1))
                    else:
                        if cache.exists():
                            try:
                                df = load_csv(str(cache))
                                data[sym] = df
                                print(
                                    f"  {YLW}{sym:<11} {len(df):>6} bars  "
                                    f"(stale cache: {df.index[-1]:%Y-%m-%d}){RST}"
                                )
                                flog.warning(
                                    "WARMUP  %-11s  %d bars  stale_cache=%s (API: %s)",
                                    sym, len(df), df.index[-1].date(), exc,
                                )
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
    effective_body_close_min = _effective_body_close_min(
        body_close_min,
        bar_completed_at(ts).time(),
            )

    sig, trigger = check_signal(
        row, prev_row, prev_day_high, prev_day_low, pdl_dirs_used,
        body_close_min=effective_body_close_min,
        trigger_name=trigger_name,
        bar_completed_time=bar_completed_at(ts).time(),
    )

    result = {
        "sym": sym,
        "sec": sec,
        "close": row["close"],
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
        result.update(price=p, atr=a, qty=q, sl=sl, tp=tp,
                      ts=ts, sl_mult=PDL_SL_MULT)

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
            bar_done = bar_completed_at(i)

            for cfg in SIGNAL_VARIANTS:
                state_key = (cfg["key"], sym)
                current_pos = current_positions.get(state_key)
                if current_pos is not None:
                    hit = _check_bracket_hit(current_pos, row["high"], row["low"])
                    if hit:
                        if hit == "SL":
                            daily_sl_counts[state_key] = daily_sl_counts.get(state_key, 0) + 1
                        del current_positions[state_key]

                if state_key in current_positions:
                    continue

                if not in_entry_window(bar_done.time()):
                    continue
                on_cooldown = state_key in cooldowns and bar_done < cooldowns[state_key]
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
                    cooldowns[state_key] = bar_done + COOLDOWN
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
                        r["qty"], r["sl"], r["tp"],
                        r.get("sl_mult", 1.0), ts=r["ts"],
                        variant=r["variant"],
                    )
                    phase = get_phase(bar_done.time())
                    log_signal(
                        sym, sec, r["signal"], r["trigger"], r["variant"], r["price"],
                        r["sl"], r["tp"], r["atr"], r["qty"],
                        ts=i, phase=phase,
                    )

    log_file = SIGNALS_DIR / f"signals_{scan_date}.csv"
    print(f"\n{CYN}── Dry run complete: {signals_found} signal(s) on {scan_date} ──{RST}")
    if signals_found:
        print(f"{DIM}  Log saved: {log_file}{RST}")


# ─── Live Loop (Phase-Based Architecture) ─────────────────────────────────


def live(capital: float, use_websocket: bool = True, auto_trade: bool = False) -> None:
    now_t = _now().time()
    shutdown_at = ENTRY_AM[1] if ENTRY_PM is None else MKT_CLOSE
    if now_t > shutdown_at:
        window = f"{ENTRY_AM[0]:%H:%M}–{shutdown_at:%H:%M}"
        print(f"\n{YLW}{BLD}Market session is over (current time: {now_t:%H:%M}).{RST}")
        print(f"{YLW}Trading window is {window}.{RST}")
        print(f"{YLW}Waiting for next trading day (warmup starts at 09:00)...{RST}\n")
        while True:
            now = _now()
            if now.weekday() < 5 and time(9, 0) <= now.time() < shutdown_at:
                break
            _time.sleep(60)
        print(f"{GRN}{BLD}Market day detected — resuming startup.{RST}\n")

    _attach_log_file()

    auto_trade_active = auto_trade and ENABLE_AUTO_TRADE

    flog.info("=" * 70)
    flog.info("SESSION START  Capital=%.0f  RR=1:%.1f  Risk=%.1f%%  ws=%s  auto_trade=%s",
              capital, RR, RISK_PCT * 100, use_websocket, auto_trade_active)
    flog.info("Stocks: %s", ", ".join(s for s, _ in STOCKS))
    flog.info("Strategy: PDL-Only (SL=%.1fx ATR)  RR=1:%.1f  MaxSL/day=%d", PDL_SL_MULT, RR, MAX_SL_PER_DAY)

    if auto_trade_active:
        print(f"{GRN}{BLD}[trade] AUTO-TRADE ENABLED — real orders will be placed{RST}")
        flog.info("AUTO-TRADE: ENABLED — bracket orders will be placed via Angel One")
    elif auto_trade and not ENABLE_AUTO_TRADE:
        print(f"{YLW}[trade] --auto-trade flag set but ENABLE_AUTO_TRADE=False in code — alert only{RST}")
        flog.warning("AUTO-TRADE: disabled by ENABLE_AUTO_TRADE constant")

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
    _startup_order_path_check(auto_trade_active)

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
    pdl_levels = fetch_prev_day_levels(stock_data, _now().date())

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
    print(f"  Filter  : First 30 min require 60% body close through PDL")
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
        f"Variant: Baseline\n"
        f"Trigger: PDL ± {PDL_PREARM_BUFFER_ATR:.1f}× ATR | Close > trigger by {PDL_CLOSE_EXT_ATR:.2f}× ATR\n"
        f"Post-10 LONG: close > trigger by {POST_10_LONG_CLOSE_EXT_ATR:.2f}× ATR\n"
        f"Post-10 SHORT: close > trigger by {POST_10_SHORT_CLOSE_EXT_ATR:.2f}× ATR | SL: {PDL_SL_MULT}× ATR | RR: 1:{RR}\n"
        f"Opening filter: first 30 min require 60% body close through PDL\n"
        f"Schedule: 09:30–13:00 | Data: {data_mode}"
    )

    cooldowns: dict[tuple[str, str], datetime] = {}
    open_positions: dict[tuple[str, str], dict] = {}
    pending_entries: dict[tuple[str, str], dict] = {}
    signal_counts: dict[tuple[str, str, str], int] = {}
    pdl_dirs_used: dict[tuple[str, str], set[str]] = {}
    daily_sl_counts: dict[tuple[str, str], int] = {}
    last_processed_bar_ts: dict[str, pd.Timestamp | None] = {
        sym: stock_data[sym].index[-1] if sym in stock_data and not stock_data[sym].empty else None
        for sym, _ in STOCKS
    }
    today_signals: list[dict] = []
    realized_pnl: float = 0.0

    saved = _load_state(_now().date())
    if saved:
        cooldowns = saved["cooldowns"]
        pdl_dirs_used = saved["pdl_dirs_used"]
        daily_sl_counts = saved["daily_sl_counts"]
        signal_counts = saved["signal_counts"]
        if saved.get("open_positions"):
            open_positions = saved["open_positions"]
        if saved.get("pending_entries"):
            pending_entries = saved["pending_entries"]
        realized_pnl = saved.get("realized_pnl", 0.0)
        flog.info("STATE RESTORED  cooldowns=%d  dirs=%d  sl_counts=%d  sig_counts=%d  positions=%d  pending=%d  pnl=₹%.2f",
                  len(cooldowns), len(pdl_dirs_used), len(daily_sl_counts), len(signal_counts),
                  len(open_positions), len(pending_entries), realized_pnl)
        print(f"  {GRN}[state] Restored session state from previous run{RST}")
        if open_positions:
            for (vk, sym), pos in open_positions.items():
                print(f"  {GRN}[state]   {sym:<11} {pos['direction']:>5}  entry ₹{pos['entry_price']:,.2f}  "
                      f"SL ₹{pos['sl']:,.2f}  TP ₹{pos['tp']:,.2f}  qty={pos.get('qty', '?')}{RST}")
                flog.info("STATE POS  %-11s  %s  entry=%.2f  SL=%.2f  TP=%.2f  sl_oid=%s  tp_oid=%s",
                          sym, pos["direction"], pos["entry_price"], pos["sl"], pos["tp"],
                          pos.get("sl_order_id"), pos.get("tp_order_id"))
        if pending_entries:
            for (vk, sym), pending in pending_entries.items():
                flog.info(
                    "STATE PENDING  %-11s  %s  order_id=%s  sl=%.2f  tp=%.2f  qty=%s",
                    sym,
                    pending.get("direction"),
                    pending.get("entry_order_id"),
                    float(pending.get("sl", 0) or 0),
                    float(pending.get("tp", 0) or 0),
                    pending.get("requested_qty"),
                )
    current_date = _now().date()
    last_phase = ""
    scan_count = 0
    last_scan_candle = -1

    try:
        while True:
            now = _now()
            t = now.time()
            phase = get_phase(t)

            # Day rollover
            if now.date() != current_date:
                flog.info("DAY ROLLOVER  %s → %s", current_date, now.date())
                current_date = now.date()
                cooldowns.clear()
                open_positions.clear()
                pending_entries.clear()
                signal_counts.clear()
                pdl_dirs_used.clear()
                daily_sl_counts.clear()
                last_processed_bar_ts = {
                    sym: stock_data[sym].index[-1] if sym in stock_data and not stock_data[sym].empty else None
                    for sym, _ in STOCKS
                }
                today_signals.clear()
                realized_pnl = 0.0
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
                    if auto_trade_active and (open_positions or pending_entries):
                        flog.info(
                            "SHUTDOWN CLEANUP  reconciling %d open position(s) and %d pending entry(ies)",
                            len(open_positions), len(pending_entries),
                        )
                        try:
                            _shutdown_book = client._conn.orderBook()
                            _shutdown_orders = {}
                            if _shutdown_book and _shutdown_book.get("status") and _shutdown_book.get("data"):
                                _shutdown_orders = {o["orderid"]: o for o in _shutdown_book["data"]}
                        except Exception as _bk_exc:
                            flog.warning("SHUTDOWN CLEANUP  orderBook failed: %s", _bk_exc)
                            _shutdown_orders = {}

                        refresh_shutdown_orders = False
                        for state_key, pending in list(pending_entries.items()):
                            op_sym = state_key[1]
                            entry_oid = pending.get("entry_order_id")
                            broker_entry = _shutdown_orders.get(entry_oid) if entry_oid else None
                            if broker_entry is None:
                                send_telegram(
                                    f"\u26a0\ufe0f *PENDING ENTRY UNRESOLVED AT SHUTDOWN* — {op_sym} {pending.get('direction', '')}\n"
                                    f"Entry order: `{entry_oid or 'unknown'}`\n"
                                    f"Broker state could not be confirmed before shutdown.\n"
                                    f"Check order book manually."
                                )
                                del pending_entries[state_key]
                                continue

                            broker_status = str(broker_entry.get("orderstatus", "")).lower()
                            broker_fill_qty = int(broker_entry.get("filledshares", 0) or 0)
                            broker_fill_px = float(broker_entry.get("averageprice", 0) or 0)

                            if broker_status in ("complete", "filled") and broker_fill_qty > 0:
                                flog.info(
                                    "SHUTDOWN LATE FILL  %-11s  order_id=%s  qty=%d  avg=%.2f",
                                    op_sym, entry_oid, broker_fill_qty, broker_fill_px,
                                )
                                late_result = _reconcile_late_filled_entry(
                                    client._conn,
                                    op_sym,
                                    pending,
                                    fill_qty=broker_fill_qty,
                                    fill_price=broker_fill_px,
                                )
                                realized_pnl += float(late_result.get("pnl_delta", 0.0) or 0.0)
                                if late_result.get("sl_hit"):
                                    daily_sl_counts[state_key] = daily_sl_counts.get(state_key, 0) + 1
                                _update_today_signal_for_order(
                                    today_signals,
                                    order_id=entry_oid,
                                    sym=op_sym,
                                    direction=pending.get("direction", ""),
                                    variant=pending.get("variant", "Baseline"),
                                    price=broker_fill_px if broker_fill_px > 0 else float(pending.get("signal_price", 0) or 0),
                                    sl=float(pending.get("sl", 0) or 0),
                                    tp=float(pending.get("tp", 0) or 0),
                                    status=f"late_fill_{late_result.get('status', 'manual')}",
                                    trigger=pending.get("trigger", "PDL_BASE"),
                                    fill_time=_now(),
                                    fallback_time=pending.get("entry_time"),
                                )
                                _update_signal_csv_row(
                                    signal_time=pending.get("entry_time"),
                                    sym=op_sym,
                                    direction=pending.get("direction", ""),
                                    variant=pending.get("variant", "Baseline"),
                                    price=broker_fill_px if broker_fill_px > 0 else float(pending.get("signal_price", 0) or 0),
                                    sl=float(pending.get("sl", 0) or 0),
                                    tp=float(pending.get("tp", 0) or 0),
                                )
                                if late_result.get("position") is not None:
                                    open_positions[state_key] = late_result["position"]
                                    refresh_shutdown_orders = True
                                del pending_entries[state_key]
                            elif broker_status in ("cancelled", "rejected"):
                                flog.info(
                                    "SHUTDOWN PENDING CLEARED  %-11s  order_id=%s  broker=%s",
                                    op_sym, entry_oid, broker_status,
                                )
                                send_telegram(
                                    f"\u23f3 *ORDER NOT FILLED* — {op_sym} {pending.get('direction', '')}\n"
                                    f"Status: `{broker_status}`\n"
                                    f"Broker confirmed no fill before shutdown."
                                )
                                _update_today_signal_for_order(
                                    today_signals,
                                    order_id=entry_oid,
                                    sym=op_sym,
                                    direction=pending.get("direction", ""),
                                    variant=pending.get("variant", "Baseline"),
                                    price=float(pending.get("signal_price", 0) or 0),
                                    sl=float(pending.get("sl", 0) or 0),
                                    tp=float(pending.get("tp", 0) or 0),
                                    status=f"not_filled_{broker_status}",
                                    trigger=pending.get("trigger", "PDL_BASE"),
                                    fill_time=_now(),
                                    fallback_time=pending.get("entry_time"),
                                )
                                del pending_entries[state_key]
                            else:
                                send_telegram(
                                    f"\u26a0\ufe0f *PENDING ENTRY UNRESOLVED AT SHUTDOWN* — {op_sym} {pending.get('direction', '')}\n"
                                    f"Entry order: `{entry_oid or 'unknown'}`\n"
                                    f"Broker status: `{broker_status}`\n"
                                    f"Check order book manually."
                                )
                                # Issue 3 fix: explicitly remove unresolved pending entries
                                # from persisted state so a next-day restart does not
                                # pick them up (the date check already handles this, but
                                # removing them here keeps the state file clean).
                                del pending_entries[state_key]

                        if refresh_shutdown_orders:
                            try:
                                _shutdown_book = client._conn.orderBook()
                                if _shutdown_book and _shutdown_book.get("status") and _shutdown_book.get("data"):
                                    _shutdown_orders = {o["orderid"]: o for o in _shutdown_book["data"]}
                            except Exception as _bk_exc:
                                flog.warning("SHUTDOWN CLEANUP  refresh orderBook failed: %s", _bk_exc)

                        for (variant_key, op_sym), op_pos in list(open_positions.items()):
                            sl_oid = op_pos.get("sl_order_id")
                            tp_oid = op_pos.get("tp_order_id")

                            broker_hit = None
                            if sl_oid and sl_oid in _shutdown_orders:
                                if _shutdown_orders[sl_oid].get("orderstatus", "").lower() in ("complete", "filled"):
                                    broker_hit = "SL"
                            if not broker_hit and tp_oid and tp_oid in _shutdown_orders:
                                if _shutdown_orders[tp_oid].get("orderstatus", "").lower() in ("complete", "filled"):
                                    broker_hit = "TP"

                            if broker_hit:
                                remaining_oid = tp_oid if broker_hit == "SL" else sl_oid
                                remaining_variety = "NORMAL" if broker_hit == "SL" else "STOPLOSS"
                                if remaining_oid:
                                    cancel_order(client._conn, remaining_oid, remaining_variety)
                                    flog.info("SHUTDOWN CANCEL  %-11s  %s order_id=%s (%s hit)",
                                              op_sym, remaining_variety, remaining_oid, broker_hit)
                                exit_price = op_pos["sl"] if broker_hit == "SL" else op_pos["tp"]
                                pos_qty = int(op_pos.get("qty", 0) or 0)
                                if op_pos["direction"] == "LONG":
                                    hit_pnl = (exit_price - op_pos["entry_price"]) * pos_qty
                                else:
                                    hit_pnl = (op_pos["entry_price"] - exit_price) * pos_qty
                                realized_pnl += hit_pnl
                                flog.info("SHUTDOWN EXIT P&L  %-11s  %s  ₹%+.2f", op_sym, broker_hit, float(hit_pnl))
                                del open_positions[(variant_key, op_sym)]
                            else:
                                sl_live = False
                                tp_live = False
                                if sl_oid and sl_oid in _shutdown_orders:
                                    sl_status = _shutdown_orders[sl_oid].get("orderstatus", "").lower()
                                    sl_live = sl_status in ("open", "trigger pending", "pending")
                                if tp_oid and tp_oid in _shutdown_orders:
                                    tp_status = _shutdown_orders[tp_oid].get("orderstatus", "").lower()
                                    tp_live = tp_status in ("open", "trigger pending", "pending")

                                if sl_live or tp_live:
                                    flog.warning(
                                        "SHUTDOWN KEEP PROTECTION  %-11s  sl=%s(live=%s)  tp=%s(live=%s) — NOT cancelling",
                                        op_sym, sl_oid, sl_live, tp_oid, tp_live,
                                    )
                                    send_telegram(
                                        f"\u26a0\ufe0f *POSITION STILL OPEN AT SHUTDOWN* — {op_sym}\n"
                                        f"Direction: `{op_pos['direction']}`\n"
                                        f"Entry: `₹{op_pos['entry_price']:,.2f}` × {op_pos.get('qty', '?')}\n"
                                        f"SL: `₹{op_pos['sl']:,.2f}` {'(live)' if sl_live else '(not live)'}\n"
                                        f"TP: `₹{op_pos['tp']:,.2f}` {'(live)' if tp_live else '(not live)'}\n"
                                        f"SL/TP orders LEFT ACTIVE on exchange.\n"
                                        f"Monitor manually or let broker handle."
                                    )

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

                    _save_state(
                        current_date,
                        cooldowns,
                        pdl_dirs_used,
                        daily_sl_counts,
                        open_positions,
                        pending_entries,
                        signal_counts,
                        today_signals,
                        realized_pnl,
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

            # Pre-market wait (sleep shorter as market open approaches)
            if phase == "PRE_MARKET":
                wait = max(int((datetime.combine(now.date(), MKT_OPEN) - now).total_seconds()), 0)
                print(f"{DIM}[{now:%H:%M}] Pre-market. Opens in {wait // 60}m {wait % 60}s...{RST}")
                if wait <= 5:
                    _time.sleep(max(wait, 0.2))
                elif wait <= 120:
                    _time.sleep(5)
                else:
                    _time.sleep(60)
                continue

            # Align to 3-min candle grid (09:15, 09:18, 09:21, ...)
            now_ts = _now()
            _base = now_ts.replace(hour=9, minute=15, second=0, microsecond=0)
            _elapsed = (now_ts - _base).total_seconds()
            _candle_idx = int(_elapsed / 180) if _elapsed >= 0 else -1

            if _candle_idx <= last_scan_candle:
                nxt = _base + timedelta(
                    seconds=(last_scan_candle + 1) * 180 + POLL_BUFFER_SEC
                )
                wait_s = max(0, (nxt - _now()).total_seconds())
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
            _wait_poll = (_poll_at - _now()).total_seconds()
            if _wait_poll > 0:
                _time.sleep(_wait_poll)
            scan_delay = max(0.0, (_now() - _poll_at).total_seconds())
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
                gf_start = _time.time()
                gf_ok = 0
                with ThreadPoolExecutor(max_workers=3) as gf_pool:
                    gf_futures = {
                        gf_pool.submit(_fetch_stock_threaded, client, sym, stock_data, now): sym
                        for sym in active_syms
                    }
                    try:
                        for fut in as_completed(gf_futures, timeout=15):
                            sym_f = gf_futures[fut]
                            try:
                                _, ok, exc = fut.result()
                                if ok:
                                    gf_ok += 1
                                elif exc:
                                    flog.warning("GAP-FILL  %-11s  %s", sym_f, exc)
                            except Exception as gf_exc:
                                flog.warning("GAP-FILL  %-11s  %s", sym_f, gf_exc)
                    except TimeoutError:
                        flog.warning("GAP-FILL  deadline exceeded (15s) — %d/%d OK", gf_ok, len(active_syms))
                flog.info("GAP-FILL  %d/%d OK in %.1fs", gf_ok, len(active_syms), _time.time() - gf_start)
                if ws_recovered and ws_feed is not None:
                    ws_feed.clear_outage()

            ws_ok = ws_feed is not None and ws_feed.is_connected
            ws_stale = ws_feed is not None and ws_feed.seconds_since_last_tick > WS_STALE_SEC

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

            # Fetch broker order book once per scan (for position tracking)
            _broker_orders: dict | None = None
            if auto_trade_active and (open_positions or pending_entries):
                try:
                    _book = client._conn.orderBook()
                    if _book and _book.get("status") and _book.get("data"):
                        _broker_orders = {o["orderid"]: o for o in _book["data"]}
                except Exception as _bk_exc:
                    flog.warning("BROKER BOOK  failed: %s", _bk_exc)

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

                            if auto_trade_active and _broker_orders is not None and state_key in pending_entries:
                                pending = pending_entries[state_key]
                                entry_oid = pending.get("entry_order_id")
                                broker_entry = _broker_orders.get(entry_oid) if entry_oid else None
                                if broker_entry is not None:
                                    broker_status = str(broker_entry.get("orderstatus", "")).lower()
                                    broker_fill_qty = int(broker_entry.get("filledshares", 0) or 0)
                                    broker_fill_px = float(broker_entry.get("averageprice", 0) or 0)
                                    if broker_status in ("complete", "filled") and broker_fill_qty > 0:
                                        flog.info(
                                            "LATE FILL DETECTED  %-11s  order_id=%s  qty=%d  avg=%.2f",
                                            sym, entry_oid, broker_fill_qty, broker_fill_px,
                                        )
                                        late_result = _reconcile_late_filled_entry(
                                            client._conn,
                                            sym,
                                            pending,
                                            fill_qty=broker_fill_qty,
                                            fill_price=broker_fill_px,
                                        )
                                        realized_pnl += float(late_result.get("pnl_delta", 0.0) or 0.0)
                                        if late_result.get("sl_hit"):
                                            daily_sl_counts[state_key] = daily_sl_counts.get(state_key, 0) + 1
                                        _update_today_signal_for_order(
                                            today_signals,
                                            order_id=entry_oid,
                                            sym=sym,
                                            direction=pending.get("direction", ""),
                                            variant=pending.get("variant", "Baseline"),
                                            price=broker_fill_px if broker_fill_px > 0 else float(pending.get("signal_price", 0) or 0),
                                            sl=float(pending.get("sl", 0) or 0),
                                            tp=float(pending.get("tp", 0) or 0),
                                            status=f"late_fill_{late_result.get('status', 'manual')}",
                                            trigger=pending.get("trigger", "PDL_BASE"),
                                            fill_time=_now(),
                                            fallback_time=pending.get("entry_time"),
                                        )
                                        _update_signal_csv_row(
                                            signal_time=pending.get("entry_time"),
                                            sym=sym,
                                            direction=pending.get("direction", ""),
                                            variant=pending.get("variant", "Baseline"),
                                            price=broker_fill_px if broker_fill_px > 0 else float(pending.get("signal_price", 0) or 0),
                                            sl=float(pending.get("sl", 0) or 0),
                                            tp=float(pending.get("tp", 0) or 0),
                                        )
                                        if late_result.get("position") is not None:
                                            open_positions[state_key] = late_result["position"]
                                        del pending_entries[state_key]
                                    elif broker_status in ("cancelled", "rejected"):
                                        flog.info(
                                            "PENDING ENTRY CLEARED  %-11s  order_id=%s  broker=%s",
                                            sym, entry_oid, broker_status,
                                        )
                                        send_telegram(
                                            f"\u23f3 *ORDER NOT FILLED* — {sym} {pending.get('direction', '')}\n"
                                            f"Status: `{broker_status}`\n"
                                            f"Broker confirmed no fill before closeout."
                                        )
                                        _update_today_signal_for_order(
                                            today_signals,
                                            order_id=entry_oid,
                                            sym=sym,
                                            direction=pending.get("direction", ""),
                                            variant=pending.get("variant", "Baseline"),
                                            price=float(pending.get("signal_price", 0) or 0),
                                            sl=float(pending.get("sl", 0) or 0),
                                            tp=float(pending.get("tp", 0) or 0),
                                            status=f"not_filled_{broker_status}",
                                            trigger=pending.get("trigger", "PDL_BASE"),
                                            fill_time=_now(),
                                            fallback_time=pending.get("entry_time"),
                                        )
                                        del pending_entries[state_key]

                            if state_key in open_positions:
                                pos = open_positions[state_key]

                                if auto_trade_active and _broker_orders is not None:
                                    broker_hit = None
                                    sl_oid = pos.get("sl_order_id")
                                    tp_oid = pos.get("tp_order_id")
                                    if sl_oid and sl_oid in _broker_orders:
                                        if _broker_orders[sl_oid].get("orderstatus", "").lower() in ("complete", "filled"):
                                            broker_hit = "SL"
                                    if not broker_hit and tp_oid and tp_oid in _broker_orders:
                                        if _broker_orders[tp_oid].get("orderstatus", "").lower() in ("complete", "filled"):
                                            broker_hit = "TP"
                                    if broker_hit:
                                        flog.info("BROKER EXIT  %-11s  %s filled on broker — cleaning up", sym, broker_hit)
                                        if broker_hit == "SL" and tp_oid:
                                            cancel_order(client._conn, tp_oid, "NORMAL")
                                        elif broker_hit == "TP" and sl_oid:
                                            cancel_order(client._conn, sl_oid, "STOPLOSS")
                                        exit_price = pos["sl"] if broker_hit == "SL" else pos["tp"]
                                        pos_qty = int(pos.get("qty", 0) or 0)
                                        if pos["direction"] == "LONG":
                                            hit_pnl = (exit_price - pos["entry_price"]) * pos_qty
                                        else:
                                            hit_pnl = (pos["entry_price"] - exit_price) * pos_qty
                                        realized_pnl += hit_pnl
                                        flog.info("BROKER EXIT P&L  %-11s  %s  ₹%+.2f", sym, broker_hit, float(hit_pnl))
                                        if broker_hit == "SL":
                                            daily_sl_counts[state_key] = daily_sl_counts.get(state_key, 0) + 1
                                        del open_positions[state_key]
                                        continue

                                hit = _check_bracket_hit(pos, bar_row["high"], bar_row["low"])
                                if hit:
                                    if auto_trade_active and _broker_orders is not None:
                                        sl_oid = pos.get("sl_order_id")
                                        tp_oid = pos.get("tp_order_id")
                                        check_oid = sl_oid if hit == "SL" else tp_oid
                                        if check_oid:
                                            broker_st = _broker_orders.get(check_oid)
                                            broker_filled = (
                                                broker_st
                                                and broker_st.get("orderstatus", "").lower() in ("complete", "filled")
                                            ) if broker_st else False
                                            if not broker_filled:
                                                flog.info(
                                                    "BRACKET DETECT  %-11s  %s on bar but broker order not filled yet — skipping cancel",
                                                    sym, hit,
                                                )
                                                pos["last_bracket_ts"] = bar_ts
                                                continue

                                    exit_price = pos["sl"] if hit == "SL" else pos["tp"]
                                    pos_qty = int(pos.get("qty", 0) or 0)
                                    if pos["direction"] == "LONG":
                                        hit_pnl = (exit_price - pos["entry_price"]) * pos_qty
                                    else:
                                        hit_pnl = (pos["entry_price"] - exit_price) * pos_qty
                                    realized_pnl += hit_pnl
                                    flog.info(
                                        "BRACKET  %-11s  %-11s  %s hit at %s — P&L ₹%+.2f  (day total ₹%+.2f)",
                                        sym, cfg["label"], hit, bar_completed_ts.strftime("%H:%M"),
                                        float(hit_pnl), float(realized_pnl),
                                    )
                                    if hit == "SL":
                                        daily_sl_counts[state_key] = daily_sl_counts.get(state_key, 0) + 1
                                        flog.info(
                                            "SL_COUNT  %-11s  %-11s  %d / %d",
                                            sym, cfg["label"], daily_sl_counts[state_key], MAX_SL_PER_DAY,
                                        )
                                        remaining_id = pos.get("tp_order_id")
                                        if remaining_id and auto_trade_active:
                                            cancel_order(client._conn, remaining_id, "NORMAL")
                                            flog.info("CANCEL TP  %-11s  order_id=%s (SL hit)", sym, remaining_id)
                                    else:
                                        remaining_id = pos.get("sl_order_id")
                                        if remaining_id and auto_trade_active:
                                            cancel_order(client._conn, remaining_id, "STOPLOSS")
                                            flog.info("CANCEL SL  %-11s  order_id=%s (TP hit)", sym, remaining_id)
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
                            raw_sig = r.get("signal") or "-"
                            raw_trig = r.get("trigger") or "-"

                            on_cooldown = state_key in cooldowns and bar_completed_ts < cooldowns[state_key]
                            entry_pending = state_key in pending_entries
                            already_in_position = state_key in open_positions
                            session_key = "AM" if bar_phase == "ACTIVE_AM" else "PM"
                            count_key = (cfg["key"], sym, session_key)
                            at_signal_cap = signal_counts.get(count_key, 0) >= MAX_SIGNALS_PER_STOCK
                            at_sl_cap = daily_sl_counts.get(state_key, 0) >= MAX_SL_PER_DAY
                            at_pos_cap = len(open_positions) >= MAX_CONCURRENT_POSITIONS
                            at_loss_cap = realized_pnl <= -MAX_DAILY_LOSS

                            if (
                                bar_is_active
                                and result["signal"]
                                and not on_cooldown
                                and not already_in_position
                                and not entry_pending
                                and not at_signal_cap
                                and not at_sl_cap
                                and not at_pos_cap
                                and not at_loss_cap
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
                                    r["atr"], r["qty"], r["sl"], r["tp"],
                                    r.get("sl_mult", 1.0), variant=r["variant"],
                                )
                                send_telegram(format_tg_signal(
                                    sym, sec, r["signal"], r["trigger"], r["variant"], r["price"],
                                    r["atr"], r["qty"], r["sl"], r["tp"],
                                    r.get("sl_mult", 1.0),
                                ))
                                log_signal(
                                    sym, sec, r["signal"], r["trigger"], r["variant"], r["price"],
                                    r["sl"], r["tp"], r["atr"], r["qty"],
                                    phase=bar_phase,
                                )

                                position_taken = not auto_trade_active
                                trade_filled = not auto_trade_active
                                signal_consumed = True
                                tracked_entry_price = r["price"]
                                tracked_qty = r["qty"]
                                logged_qty = r["qty"]
                                position_note = ""
                                tracked_order_id = None

                                if auto_trade_active:
                                    position_taken = False
                                    trade_filled = False
                                    bar_close_f = float(r["close"])
                                    entry_buf = _entry_buffer(r["atr"]) if r["atr"] > 0 else 0.10
                                    if r["signal"] == "LONG":
                                        planned_entry_price = round_to_tick(bar_close_f + entry_buf)
                                        remaining_reward = r["tp"] - planned_entry_price
                                    else:
                                        planned_entry_price = round_to_tick(bar_close_f - entry_buf)
                                        remaining_reward = planned_entry_price - r["tp"]
                                    min_reward = r["atr"] * MIN_REMAINING_REWARD_ATR
                                    if remaining_reward < min_reward:
                                        position_note = (
                                            f"skip_low_reward("
                                            f"remaining={remaining_reward:.2f},"
                                            f"min={min_reward:.2f},"
                                            f"entry={planned_entry_price:.2f},"
                                            f"tp={r['tp']:.2f})"
                                        )
                                        flog.warning(
                                            "LOW REWARD SKIP  %-11s  %s  remaining=%.2f < min=%.2f (%.1f%% ATR)  "
                                            "planned_entry=%.2f  tp=%.2f — entry too close to TP, skipping order",
                                            sym, r["signal"], remaining_reward, min_reward,
                                            MIN_REMAINING_REWARD_ATR * 100, planned_entry_price, r["tp"],
                                        )
                                        send_telegram(
                                            f"\u23ed\ufe0f *SKIP LOW REWARD* — {sym} {r['signal']}\n"
                                            f"Planned entry: `₹{planned_entry_price:,.2f}` | TP: `₹{r['tp']:,.2f}`\n"
                                            f"Remaining: `₹{remaining_reward:,.2f}` < min `₹{min_reward:,.2f}` "
                                            f"({MIN_REMAINING_REWARD_ATR:.0%} ATR)\n"
                                            f"Signal logged but order skipped"
                                        )
                                        order_result = None
                                    else:
                                        order_result = place_bracket_order(
                                            client._conn, sym, r["signal"],
                                            r["price"], r["sl"], r["tp"], r["qty"],
                                            atr=r["atr"],
                                            bar_close=bar_close_f,
                                        )
                                    if order_result is not None and order_result["success"]:
                                        tracked_order_id = order_result["order_id"]
                                        next_bar_expiry = (
                                            bar_completed_ts + BAR_INTERVAL
                                        ).to_pydatetime() if hasattr(bar_completed_ts, 'to_pydatetime') else None
                                        fill_result = wait_for_fill_or_cancel(
                                            client._conn, sym, order_result["order_id"],
                                            bar_expiry_ts=next_bar_expiry,
                                        )
                                        if fill_result["filled"]:
                                            fill_qty = fill_result["filled_qty"]
                                            fill_px = fill_result["fill_price"]
                                            tracked_qty = fill_qty if fill_qty > 0 else r["qty"]
                                            tracked_entry_price = fill_px if fill_px > 0 else r["price"]
                                            logged_qty = tracked_qty
                                            fill_is_partial = tracked_qty < r["qty"]
                                            fill_ratio = tracked_qty / max(int(r["qty"]) or 1, 1)
                                            actual_gross_target = gross_target_rupees(
                                                r["signal"], tracked_entry_price, r["tp"], tracked_qty,
                                            )

                                            keep_partial = (
                                                not fill_is_partial
                                                or (
                                                    fill_ratio >= MIN_FILL_RATIO
                                                    and actual_gross_target >= MIN_ACTUAL_GROSS_TARGET
                                                )
                                            )

                                            if fill_is_partial and not keep_partial:
                                                position_note = (
                                                    "partial_rejected("
                                                    f"{tracked_qty}/{r['qty']},"
                                                    f"ratio={fill_ratio:.0%},"
                                                    f"tp=₹{actual_gross_target:,.0f})"
                                                )
                                                flog.warning(
                                                    "PARTIAL FILL REJECTED  %-11s  %d/%d (%.0f%%)  "
                                                    "gross_tp=₹%.2f  thresholds=%.0f%%/₹%.2f",
                                                    sym, tracked_qty, r["qty"], fill_ratio * 100.0,
                                                    actual_gross_target,
                                                    MIN_FILL_RATIO * 100.0, MIN_ACTUAL_GROSS_TARGET,
                                                )
                                                exit_result = place_market_exit_order(
                                                    client._conn, sym, r["signal"], tracked_qty,
                                                    ref_price=tracked_entry_price,
                                                )
                                                if exit_result["success"]:
                                                    flat_result = wait_for_fill_or_cancel(
                                                        client._conn,
                                                        sym,
                                                        exit_result["order_id"],
                                                        timeout_sec=PARTIAL_EXIT_TIMEOUT_SEC,
                                                    )
                                                    exit_filled_qty = int(flat_result.get("filled_qty", 0) or 0)
                                                    exit_fill_px = float(flat_result.get("fill_price", 0) or 0)
                                                    if flat_result["filled"] and exit_filled_qty > 0:
                                                        flattened_qty = min(exit_filled_qty, tracked_qty)
                                                        if r["signal"] == "LONG":
                                                            flat_pnl = (
                                                                exit_fill_px - tracked_entry_price
                                                            ) * flattened_qty
                                                        else:
                                                            flat_pnl = (
                                                                tracked_entry_price - exit_fill_px
                                                            ) * flattened_qty
                                                        realized_pnl += flat_pnl
                                                        if flattened_qty >= tracked_qty:
                                                            signal_consumed = False
                                                            send_telegram(
                                                                f"\u26a0\ufe0f *PARTIAL FILL REJECTED* — {sym} {r['signal']}\n"
                                                                f"Entry: `{order_result['order_id']}` @ ₹{tracked_entry_price:,.2f} × "
                                                                f"{tracked_qty}/{r['qty']}\n"
                                                                f"Fill ratio: `{fill_ratio:.0%}` | Gross TP: `₹{actual_gross_target:,.2f}`\n"
                                                                f"Flattened: `{exit_result['order_id']}` @ ₹{exit_fill_px:,.2f} × {flattened_qty}\n"
                                                                f"Realized: `₹{flat_pnl:+,.2f}`"
                                                            )
                                                            tracked_qty = flattened_qty
                                                            logged_qty = flattened_qty
                                                        else:
                                                            tracked_qty -= flattened_qty
                                                            logged_qty = tracked_qty
                                                            actual_gross_target = gross_target_rupees(
                                                                r["signal"], tracked_entry_price, r["tp"], tracked_qty,
                                                            )
                                                            position_note = (
                                                                "partial_exit_incomplete("
                                                                f"remaining={tracked_qty}/{r['qty']},"
                                                                f"tp=₹{actual_gross_target:,.0f})"
                                                            )
                                                            flog.warning(
                                                                "PARTIAL EXIT INCOMPLETE  %-11s  exited=%d  remaining=%d",
                                                                sym, flattened_qty, tracked_qty,
                                                            )
                                                            send_telegram(
                                                                f"\u26a0\ufe0f *PARTIAL FILL BELOW THRESHOLD* — {sym} {r['signal']}\n"
                                                                f"Entry: `{order_result['order_id']}` @ ₹{tracked_entry_price:,.2f}\n"
                                                                f"Rejected fill: `{flattened_qty + tracked_qty}/{r['qty']}` | "
                                                                f"Flattened: `{flattened_qty}`\n"
                                                                f"Remaining protected qty: `{tracked_qty}`"
                                                            )
                                                            keep_partial = True
                                                    else:
                                                        position_note = (
                                                            "partial_exit_failed("
                                                            f"{tracked_qty}/{r['qty']})"
                                                        )
                                                        flog.error(
                                                            "PARTIAL EXIT FAILED  %-11s  flatten order did not fill (%s)",
                                                            sym, flat_result.get("status", "unknown"),
                                                        )
                                                        send_telegram(
                                                            f"\u26a0\ufe0f *PARTIAL FILL BELOW THRESHOLD* — {sym} {r['signal']}\n"
                                                            f"Entry: `{order_result['order_id']}` @ ₹{tracked_entry_price:,.2f} × "
                                                            f"{tracked_qty}/{r['qty']}\n"
                                                            f"Flatten attempt failed: `{flat_result.get('status', 'unknown')}`\n"
                                                            f"Keeping remaining qty protected with SL/TP"
                                                        )
                                                        keep_partial = True
                                                else:
                                                    position_note = (
                                                        "partial_exit_order_failed("
                                                        f"{tracked_qty}/{r['qty']})"
                                                    )
                                                    flog.error(
                                                        "PARTIAL EXIT ORDER FAILED  %-11s  %s",
                                                        sym, exit_result["message"],
                                                    )
                                                    send_telegram(
                                                        f"\u26a0\ufe0f *PARTIAL FILL BELOW THRESHOLD* — {sym} {r['signal']}\n"
                                                        f"Entry: `{order_result['order_id']}` @ ₹{tracked_entry_price:,.2f} × "
                                                        f"{tracked_qty}/{r['qty']}\n"
                                                        f"Flatten order failed: `{exit_result['message']}`\n"
                                                        f"Keeping remaining qty protected with SL/TP"
                                                    )
                                                    keep_partial = True

                                            if keep_partial and tracked_qty > 0:
                                                sl_result = place_sl_order(
                                                    client._conn, sym, r["signal"],
                                                    r["sl"], tracked_qty,
                                                )
                                                tp_result = place_tp_order(
                                                    client._conn, sym, r["signal"],
                                                    r["tp"], tracked_qty,
                                                )
                                                sl_msg = f"\nSL: `{sl_result['order_id']}`" if sl_result["success"] else f"\nSL FAILED: {sl_result['message']}"
                                                tp_msg = f"\nTP: `{tp_result['order_id']}`" if tp_result["success"] else f"\nTP FAILED: {tp_result['message']}"
                                                r["_sl_order_id"] = sl_result.get("order_id")
                                                r["_tp_order_id"] = tp_result.get("order_id")
                                                if sl_result["success"] and tp_result["success"]:
                                                    fill_title = (
                                                        "\u2705 *ORDER PARTIAL FILLED*"
                                                        if fill_is_partial
                                                        else "\u2705 *ORDER FILLED*"
                                                    )
                                                    fill_qty_msg = (
                                                        f"{tracked_qty}/{r['qty']}"
                                                        if fill_is_partial
                                                        else f"{tracked_qty}"
                                                    )
                                                    send_telegram(
                                                        f"{fill_title} — {sym} {r['signal']}\n"
                                                        f"Entry: `{order_result['order_id']}` @ ₹{tracked_entry_price:,.2f} × {fill_qty_msg}\n"
                                                        f"SL: ₹{r['sl']:,.2f} | TP: ₹{r['tp']:,.2f}"
                                                        f"{sl_msg}{tp_msg}"
                                                    )
                                                    trade_filled = True
                                                    position_taken = True

                                                    _imm_sl_oid = sl_result.get("order_id")
                                                    _imm_tp_oid = tp_result.get("order_id")
                                                    try:
                                                        _imm_book = client._conn.orderBook()
                                                        _imm_orders = (
                                                            {o["orderid"]: o for o in _imm_book["data"]}
                                                            if _imm_book and _imm_book.get("data")
                                                            else {}
                                                        )
                                                        _imm_sl_filled = (
                                                            _imm_sl_oid
                                                            and _imm_sl_oid in _imm_orders
                                                            and _imm_orders[_imm_sl_oid].get("orderstatus", "").lower()
                                                            in ("complete", "filled")
                                                        )
                                                        _imm_tp_filled = (
                                                            _imm_tp_oid
                                                            and _imm_tp_oid in _imm_orders
                                                            and _imm_orders[_imm_tp_oid].get("orderstatus", "").lower()
                                                            in ("complete", "filled")
                                                        )
                                                        if _imm_tp_filled and _imm_sl_oid:
                                                            cancel_order(client._conn, _imm_sl_oid, "STOPLOSS")
                                                            flog.info(
                                                                "IMM RECONCILE  %-11s  TP already filled — cancelled SL %s",
                                                                sym, _imm_sl_oid,
                                                            )
                                                            position_note = "immediate_tp"
                                                            exit_price = r["tp"]
                                                            pos_qty = tracked_qty
                                                            if r["signal"] == "LONG":
                                                                hit_pnl = (exit_price - tracked_entry_price) * pos_qty
                                                            else:
                                                                hit_pnl = (tracked_entry_price - exit_price) * pos_qty
                                                            realized_pnl += hit_pnl
                                                            flog.info("IMM RECONCILE P&L  %-11s  TP  ₹%+.2f", sym, float(hit_pnl))
                                                            position_taken = False
                                                        elif _imm_sl_filled and _imm_tp_oid:
                                                            cancel_order(client._conn, _imm_tp_oid, "NORMAL")
                                                            flog.info(
                                                                "IMM RECONCILE  %-11s  SL already filled — cancelled TP %s",
                                                                sym, _imm_tp_oid,
                                                            )
                                                            position_note = "immediate_sl"
                                                            exit_price = r["sl"]
                                                            pos_qty = tracked_qty
                                                            if r["signal"] == "LONG":
                                                                hit_pnl = (exit_price - tracked_entry_price) * pos_qty
                                                            else:
                                                                hit_pnl = (tracked_entry_price - exit_price) * pos_qty
                                                            realized_pnl += hit_pnl
                                                            daily_sl_counts[state_key] = daily_sl_counts.get(state_key, 0) + 1
                                                            flog.info("IMM RECONCILE P&L  %-11s  SL  ₹%+.2f", sym, float(hit_pnl))
                                                            position_taken = False
                                                    except Exception as _imm_exc:
                                                        flog.warning("IMM RECONCILE  %-11s  check failed: %s", sym, _imm_exc)

                                                else:
                                                    position_note = (
                                                        "protection_failed("
                                                        f"sl={'ok' if sl_result['success'] else 'fail'},"
                                                        f"tp={'ok' if tp_result['success'] else 'fail'})"
                                                    )
                                                    flog.error(
                                                        "PROTECTION FAILED  %-11s  sl=%s  tp=%s — attempting emergency flatten",
                                                        sym, sl_result["success"], tp_result["success"],
                                                    )
                                                    emergency_exit = place_market_exit_order(
                                                        client._conn, sym, r["signal"], tracked_qty,
                                                        ref_price=tracked_entry_price,
                                                    )
                                                    exit_status = "order_failed"
                                                    exit_fill_px = 0.0
                                                    flattened_qty = 0
                                                    if emergency_exit["success"]:
                                                        flat_result = wait_for_fill_or_cancel(
                                                            client._conn,
                                                            sym,
                                                            emergency_exit["order_id"],
                                                            timeout_sec=PARTIAL_EXIT_TIMEOUT_SEC,
                                                        )
                                                        exit_status = flat_result.get("status", "unknown")
                                                        exit_fill_px = float(flat_result.get("fill_price", 0) or 0)
                                                        flattened_qty = min(
                                                            int(flat_result.get("filled_qty", 0) or 0),
                                                            tracked_qty,
                                                        )
                                                        if flattened_qty > 0:
                                                            if r["signal"] == "LONG":
                                                                flat_pnl = (
                                                                    exit_fill_px - tracked_entry_price
                                                                ) * flattened_qty
                                                            else:
                                                                flat_pnl = (
                                                                    tracked_entry_price - exit_fill_px
                                                                ) * flattened_qty
                                                            realized_pnl += flat_pnl
                                                        if flat_result["filled"] and flattened_qty >= tracked_qty:
                                                            if sl_result["success"] and r.get("_sl_order_id"):
                                                                cancel_order(client._conn, r["_sl_order_id"], "STOPLOSS")
                                                            if tp_result["success"] and r.get("_tp_order_id"):
                                                                cancel_order(client._conn, r["_tp_order_id"], "NORMAL")
                                                            logged_qty = flattened_qty
                                                            send_telegram(
                                                                f"\u26a0\ufe0f *PROTECTION FAILED — POSITION FLATTENED* — {sym} {r['signal']}\n"
                                                                f"Entry: `{order_result['order_id']}` @ ₹{tracked_entry_price:,.2f} × {flattened_qty}\n"
                                                                f"SL status: `{'ok' if sl_result['success'] else 'failed'}` | "
                                                                f"TP status: `{'ok' if tp_result['success'] else 'failed'}`\n"
                                                                f"Emergency exit: `{emergency_exit['order_id']}` @ ₹{exit_fill_px:,.2f}\n"
                                                                f"Realized: `₹{flat_pnl:+,.2f}`"
                                                            )
                                                        else:
                                                            remaining_qty = max(tracked_qty - flattened_qty, 0)
                                                            logged_qty = remaining_qty or tracked_qty
                                                            send_telegram(
                                                                f"\u26a0\ufe0f *MANUAL ACTION REQUIRED* — {sym} {r['signal']}\n"
                                                                f"Protective order placement failed.\n"
                                                                f"SL status: `{'ok' if sl_result['success'] else 'failed'}` | "
                                                                f"TP status: `{'ok' if tp_result['success'] else 'failed'}`\n"
                                                                f"Emergency exit status: `{exit_status}`\n"
                                                                f"Filled to flatten: `{flattened_qty}/{tracked_qty}`\n"
                                                                f"Remaining qty may still be live. Check broker immediately."
                                                            )
                                                    else:
                                                        send_telegram(
                                                            f"\u26a0\ufe0f *MANUAL ACTION REQUIRED* — {sym} {r['signal']}\n"
                                                            f"Protective order placement failed and emergency exit order was rejected.\n"
                                                            f"SL status: `{'ok' if sl_result['success'] else 'failed'}` | "
                                                            f"TP status: `{'ok' if tp_result['success'] else 'failed'}`\n"
                                                            f"Exit error: `{emergency_exit['message']}`\n"
                                                            f"Check broker immediately."
                                                        )
                                        else:
                                            fill_status = fill_result.get("status", "unknown")
                                            if fill_result.get("resolved", False) and fill_status in ("cancelled", "rejected"):
                                                position_note = f"entry_not_filled({fill_status})"
                                                flog.warning(
                                                    "POSITION SKIP  %-11s  entry order not filled (%s) — "
                                                    "signal logged but no open position recorded",
                                                    sym, fill_status,
                                                )
                                                send_telegram(
                                                    f"\u23f3 *ORDER NOT FILLED* — {sym} {r['signal']}\n"
                                                    f"Status: `{fill_status}`\n"
                                                    f"Broker confirmed no fill before closeout."
                                                )
                                            else:
                                                position_note = f"entry_pending({fill_status})"
                                                pending_entries[state_key] = {
                                                    "direction": r["signal"],
                                                    "variant": r["variant"],
                                                    "trigger": r["trigger"],
                                                    "requested_qty": r["qty"],
                                                    "signal_price": r["price"],
                                                    "sl": r["sl"],
                                                    "tp": r["tp"],
                                                    "atr": r["atr"],
                                                    "entry_order_id": order_result["order_id"],
                                                    "entry_time": bar_completed_ts,
                                                    "bar_ts": bar_ts,
                                                    "entry_scan": scan_count,
                                                }
                                                flog.warning(
                                                    "ENTRY PENDING  %-11s  order_id=%s  unresolved after timeout/cancel (%s) — monitoring broker",
                                                    sym, order_result["order_id"], fill_status,
                                                )
                                                send_telegram(
                                                    f"\u23f3 *ENTRY STATUS PENDING* — {sym} {r['signal']}\n"
                                                    f"Entry order: `{order_result['order_id']}`\n"
                                                    f"Broker status after timeout/cancel: `{fill_status}`\n"
                                                    f"Bot will keep monitoring and will auto-protect the trade if it fills late."
                                                )
                                    elif order_result is not None:
                                        position_note = "entry_order_failed"
                                        flog.error("AUTO-TRADE FAILED  %-11s  %s", sym, order_result["message"])
                                        flog.warning(
                                            "POSITION SKIP  %-11s  entry order failed — "
                                            "signal logged but no open position recorded",
                                            sym,
                                        )
                                        send_telegram(
                                            f"\u274c *ORDER FAILED* — {sym} {r['signal']}\n"
                                            f"Error: {order_result['message']}\n"
                                            f"Manual entry needed: ₹{r['price']:,.2f}"
                                        )

                                if signal_consumed:
                                    cooldowns[state_key] = bar_completed_ts + COOLDOWN
                                    signal_counts[count_key] = signal_counts.get(count_key, 0) + 1
                                    sym_pdl_dirs.add(r["signal"])

                                if position_taken:
                                    new_pos = {
                                        "direction": r["signal"],
                                        "entry_price": tracked_entry_price,
                                        "entry_atr": r["atr"],
                                        "qty": tracked_qty,
                                        "sl": r["sl"],
                                        "tp": r["tp"],
                                        "entry_scan": scan_count,
                                        "entry_time": bar_completed_ts,
                                        "last_bracket_ts": bar_ts,
                                        "variant": r["variant"],
                                        "sl_order_id": r.get("_sl_order_id"),
                                        "tp_order_id": r.get("_tp_order_id"),
                                    }
                                    same_bar_hit = _check_bracket_hit(new_pos, bar_row["high"], bar_row["low"])
                                    if same_bar_hit:
                                        flog.warning(
                                            "SAME-BAR EXIT  %-11s  %s on entry bar — %s",
                                            sym, r["signal"], same_bar_hit,
                                        )
                                        if auto_trade_active:
                                            flog.info(
                                                "SAME-BAR SKIP  %-11s  auto-trade active — "
                                                "broker handles SL/TP, keeping position tracked",
                                                sym,
                                            )
                                            open_positions[state_key] = new_pos
                                        else:
                                            if same_bar_hit == "SL":
                                                daily_sl_counts[state_key] = daily_sl_counts.get(state_key, 0) + 1
                                    else:
                                        open_positions[state_key] = new_pos

                                if signal_consumed or position_taken or trade_filled:
                                    today_signals.append({
                                        "sym": sym,
                                        "dir": r["signal"],
                                        "trigger": r["trigger"],
                                        "variant": r["variant"],
                                        "price": tracked_entry_price if trade_filled else r["price"],
                                        "sl": r["sl"],
                                        "tp": r["tp"],
                                        "time": now,
                                        "status": "filled" if trade_filled else "signal_only",
                                        "order_id": tracked_order_id,
                                    })

                                    flog.info(
                                        "  %-11s  %-11s  C=%.2f  → ★ %s[%s]  SL=%.2f  TP=%.2f  Qty=%d%s",
                                        sym, r["variant"], r["close"],
                                        r["signal"], r["trigger"], r["sl"], r["tp"],
                                        logged_qty,
                                        f"  {position_note}" if position_note else "",
                                    )
                            else:
                                reason = ""
                                if not bar_is_active:
                                    reason = "inactive_phase"
                                elif entry_pending:
                                    reason = "entry_pending"
                                elif already_in_position:
                                    reason = "in_position"
                                elif on_cooldown:
                                    reason = f"cooldown_until_{cooldowns[state_key]:%H:%M}"
                                elif at_signal_cap:
                                    reason = f"max_signals_{session_key}({signal_counts.get(count_key, 0)})"
                                elif at_sl_cap:
                                    reason = f"sl_cap({daily_sl_counts.get(state_key, 0)}/{MAX_SL_PER_DAY})"
                                elif at_pos_cap:
                                    reason = f"max_positions({len(open_positions)}/{MAX_CONCURRENT_POSITIONS})"
                                elif at_loss_cap:
                                    reason = f"daily_loss_cap(₹{realized_pnl:+,.0f})"
                                elif raw_sig != "-":
                                    reason = "blocked"
                                result["signal"] = None
                                flog.debug(
                                    "  %-11s  %-11s  C=%.2f  → %s[%s]  %s",
                                    sym, r["variant"], r["close"],
                                    raw_sig, raw_trig, reason,
                                )

                    last_processed_bar_ts[sym] = latest_pending_ts

                if latest_dashboard_result is not None:
                    dashboard.append(latest_dashboard_result)

            scan_elapsed = _time.time() - scan_start
            flog.info("SCAN #%d complete in %.1fs", scan_count, scan_elapsed)
            print_dashboard(dashboard, phase)
            _save_state(current_date, cooldowns, pdl_dirs_used, daily_sl_counts,
                        open_positions, pending_entries, signal_counts, today_signals,
                        realized_pnl)

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
        "--capital", type=float, default=25_000,
        help="Trading capital in INR (default: 25000)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Scan last cached session for signals, then exit",
    )
    parser.add_argument(
        "--no-websocket", action="store_true",
        help="Disable WebSocket feed, use REST polling only",
    )
    parser.add_argument(
        "--auto-trade", action="store_true",
        help="Enable automatic bracket order placement via Angel One",
    )
    args = parser.parse_args()

    print(f"\n{BLD}NSE Intraday Scalping — Live Signal Generator{RST}")
    print(f"{DIM}Prearmed PDL ({PDL_PREARM_BUFFER_ATR:.1f}×ATR trigger buffer, SL {PDL_SL_MULT}×ATR)")
    print(f"RR 1:{RR} | Risk {RISK_PCT * 100:.1f}% | Max {MAX_SL_PER_DAY} SL/stock/day | Close > trigger by {PDL_CLOSE_EXT_ATR:.2f} ATR")
    print(f"{DIM}Post-10 LONG close > trigger by {POST_10_LONG_CLOSE_EXT_ATR:.2f} ATR | Post-10 SHORT close > trigger by {POST_10_SHORT_CLOSE_EXT_ATR:.2f} ATR | First 30 min need 60% close{RST}\n")

    if args.dry_run:
        dry_run(args.capital)
    else:
        try:
            live(args.capital, use_websocket=not args.no_websocket,
                 auto_trade=args.auto_trade)
        except KeyboardInterrupt:
            flog.info("SESSION END  user interrupt")
            print(f"\n{YLW}Session ended by user.{RST}")
        except Exception as exc:
            flog.error("SESSION CRASH  %s: %s", type(exc).__name__, exc)
            print(f"\n{RED}CRASH: {exc}{RST}")
            raise


if __name__ == "__main__":
    main()

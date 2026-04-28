"""
MCX Silver Micro — Live Signal Monitor (PDL Breakout)

Streams 3-min candles via Angel One REST API, computes ATR-based PDL
breakout signals, and sends alerts to console + Telegram.

OBSERVATION-FIRST:  No orders are placed by default.  Watch signals for
a few weeks, track virtual P&L, and switch to auto-trade only once you
have confidence in the setup.

STRATEGY (same core as live_signals.py):
  PDL Breakout — trigger placed just beyond yesterday's High/Low.
  Buffer = 0.1× ATR.  Signal fires when a completed 3-min bar trades
  through the trigger and still closes through the base prev-day level.

MCX Session Phases:
  Phase 1  17:45  Warm-Up    — connect, fetch 5 days of 3-min data
  Phase 2  18:00  Active     — evening session, signals ON
  Phase 3  22:30  Wind-Down  — no new entries, track open virtual position
  Phase 4  23:10  Shutdown   — force virtual exit, end session

Contract: Front-month SILVERMIC (1 kg lot), roll 5 days before expiry.

Usage:
    python silver_micro_signals.py                  # default ₹50,000
    python silver_micro_signals.py --capital 50000
    python silver_micro_signals.py --dry-run
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import time as _time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from fetch_data import AngelOneClient

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))


# ─── Contract Configuration ──────────────────────────────────────────────

EXCHANGE = "MCX"
SYMBOL_ROOT = "SILVERMIC"
LOT_SIZE = 1          # kg
TICK_SIZE = 1.0        # ₹1 per kg
MARGIN_PCT = 0.1275    # ~12.75% normal margin
ROLL_DAYS_BEFORE_EXPIRY = 5

# ─── Strategy Parameters ─────────────────────────────────────────────────

ATR_PERIOD = 14
RR = 1.5
PDL_PREARM_BUFFER_ATR = 0.10
PDL_SL_MULT = 1.0
MAX_SL_PER_DAY = 2
MAX_SL_RISK_PER_TRADE = 1500.0   # skip if SL distance × lot > ₹1,500
MAX_DAILY_LOSS = 2000.0
MAX_WEEKLY_LOSS = 4000.0

COOLDOWN = timedelta(minutes=30)
MAX_SIGNALS_PER_SESSION = 2
BAR_INTERVAL = timedelta(minutes=3)
WARMUP_3MIN_DAYS = 5
POLL_BUFFER_SEC = 2

# ─── Time Rules (MCX) ────────────────────────────────────────────────────

MCX_OPEN = time(9, 0)
MCX_CLOSE = time(23, 30)
ENTRY_START = time(15, 0)
ENTRY_END = time(22, 30)
FORCE_EXIT_TIME = time(23, 10)

# ─── File Paths ──────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"
LOG_DIR = Path(__file__).parent / "logs"
LIVE_LOG_DIR = LOG_DIR / "silver_micro"
SIGNALS_DIR = LOG_DIR / "signals_mcx"
STATE_FILE = LOG_DIR / ".silver_micro_state.json"
WEEKLY_FILE = LOG_DIR / ".silver_micro_weekly.json"

ENABLE_AUTO_TRADE = False

IST = ZoneInfo("Asia/Kolkata")


def _now() -> datetime:
    return datetime.now(IST).replace(tzinfo=None)


# ─── Logging ─────────────────────────────────────────────────────────────

logging.getLogger().handlers = []
logging.getLogger().addHandler(logging.NullHandler())

for _lib_name in ("smartConnect", "SmartApi", "logzero", "SmartWebSocketV2",
                   "websocket", "urllib3"):
    _lib_log = logging.getLogger(_lib_name)
    _lib_log.setLevel(logging.CRITICAL)
    _lib_log.handlers = []
    _lib_log.addHandler(logging.NullHandler())


def _init_file_logger() -> logging.Logger:
    logger = logging.getLogger("silver_micro")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger


def _attach_log_file():
    LIVE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    flog.handlers = [h for h in flog.handlers if not isinstance(h, logging.FileHandler)]
    fh = logging.FileHandler(
        LIVE_LOG_DIR / f"silver_{_now():%Y-%m-%d}.log", encoding="utf-8",
    )
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
    flog.addHandler(fh)


flog = _init_file_logger()


# ─── Terminal Colors ─────────────────────────────────────────────────────

GRN, RED, YLW, CYN = "\033[92m", "\033[91m", "\033[93m", "\033[96m"
BLD, DIM, RST = "\033[1m", "\033[2m", "\033[0m"


# ─── Telegram ────────────────────────────────────────────────────────────

TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")


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


# ─── Contract Resolution ─────────────────────────────────────────────────


def resolve_front_month(client: AngelOneClient) -> tuple[str, str]:
    """Find the nearest SILVERMIC FUT contract with at least ROLL_DAYS_BEFORE_EXPIRY remaining.

    Returns (tradingsymbol, symboltoken).
    """
    resp = client._conn.searchScrip(EXCHANGE, SYMBOL_ROOT)
    if not resp or not resp.get("status") or not resp.get("data"):
        raise RuntimeError(f"searchScrip({EXCHANGE}, {SYMBOL_ROOT}) returned no results")

    month_map = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }
    today = _now().date()
    min_expiry = today + timedelta(days=ROLL_DAYS_BEFORE_EXPIRY)
    candidates = []

    for entry in resp["data"]:
        tsym = entry.get("tradingsymbol", "")
        token = entry.get("symboltoken", "")
        if not tsym.startswith(SYMBOL_ROOT) or "FUT" not in tsym:
            continue
        m = re.search(r"(\d{1,2})([A-Z]{3})(\d{2})FUT$", tsym)
        if not m:
            continue
        day, mon_str, yr = int(m.group(1)), m.group(2), int(m.group(3)) + 2000
        mon = month_map.get(mon_str)
        if mon is None:
            continue
        try:
            expiry = date(yr, mon, day)
        except ValueError:
            continue
        if expiry >= min_expiry:
            candidates.append((expiry, tsym, token))

    if not candidates:
        raise RuntimeError(
            f"No SILVERMIC FUT contract with expiry >= {min_expiry}. "
            "Check Angel One instrument master."
        )

    candidates.sort()
    expiry, tsym, token = candidates[0]
    return tsym, token


# ─── MCX Data Fetching ───────────────────────────────────────────────────


def fetch_mcx_candles(
    client: AngelOneClient,
    token: str,
    from_date: str,
    to_date: str,
    retries: int = 5,
) -> pd.DataFrame:
    """Fetch 3-min candles from MCX via getCandleData with retry logic."""
    api_timeout = 15

    for attempt in range(retries):
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(
                    client._conn.getCandleData,
                    {
                        "exchange": EXCHANGE,
                        "symboltoken": token,
                        "interval": "THREE_MINUTE",
                        "fromdate": from_date,
                        "todate": to_date,
                    },
                )
                resp = fut.result(timeout=api_timeout)
        except FuturesTimeoutError:
            if attempt < retries - 1:
                _time.sleep(2.0 * (attempt + 1))
                continue
            raise RuntimeError(f"MCX candle fetch timed out after {api_timeout}s")
        except Exception as exc:
            raise RuntimeError(f"MCX candle fetch failed: {exc}")

        if resp.get("status") and resp.get("data"):
            break

        msg = str(resp.get("message", "empty"))

        if resp.get("status") and not resp.get("data"):
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"]
            ).rename_axis("datetime")

        if "TooManyRequests" in msg or "access rate" in msg.lower():
            if attempt < retries - 1:
                _time.sleep(3.0 * (attempt + 1))
                continue

        if "Invalid Token" in msg:
            raise RuntimeError(f"Invalid token {token} — contract may have expired")

        raise RuntimeError(f"MCX candle fetch failed: {msg}")

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


def fetch_warmup(
    client: AngelOneClient,
    token: str,
    days: int = WARMUP_3MIN_DAYS,
) -> pd.DataFrame:
    """Fetch multi-day 3-min data for warmup (respects API's 5-day chunk limit)."""
    chunks: list[pd.DataFrame] = []
    end = _now()
    remaining = days

    while remaining > 0:
        chunk_days = min(remaining, 5)
        start = end - timedelta(days=chunk_days)
        from_str = start.strftime("%Y-%m-%d 09:00")
        to_str = end.strftime("%Y-%m-%d 23:30")

        try:
            chunk = fetch_mcx_candles(client, token, from_str, to_str, retries=3)
            if not chunk.empty:
                chunks.append(chunk)
        except Exception as exc:
            flog.warning("WARMUP chunk %s→%s failed: %s", from_str[:10], to_str[:10], exc)

        end = start
        remaining -= chunk_days
        if remaining > 0:
            _time.sleep(0.5)

    if not chunks:
        raise RuntimeError("No warmup data fetched")

    df = pd.concat(chunks)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _cache_path() -> Path:
    return DATA_DIR / "SILVERMIC_3min.csv"


def load_warmup_cached(client: AngelOneClient, token: str) -> pd.DataFrame:
    """Load warmup data: prefer fresh API data, fall back to cache."""
    cache = _cache_path()
    today = _now().date()

    if cache.exists():
        try:
            df = pd.read_csv(cache, index_col="datetime", parse_dates=True)
            if df.index[-1].date() >= today or client is None:
                print(f"  SILVERMIC  {len(df):>6} bars  (cache: {df.index[-1]:%Y-%m-%d})")
                flog.info("WARMUP  SILVERMIC  %d bars  cache=%s", len(df), df.index[-1].date())
                return df
        except Exception:
            pass

    for attempt in range(3):
        try:
            df = fetch_warmup(client, token)
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache)
            print(
                f"  SILVERMIC  {len(df):>6} bars  "
                f"(API: {df.index[0]:%m-%d} → {df.index[-1]:%m-%d %H:%M})"
            )
            flog.info(
                "WARMUP  SILVERMIC  %d bars  API %s→%s  cached",
                len(df), df.index[0].date(), df.index[-1],
            )
            return df
        except Exception as exc:
            if attempt < 2:
                flog.warning("WARMUP  attempt %d failed: %s — retrying", attempt + 1, exc)
                _time.sleep(3.0 * (attempt + 1))

    if cache.exists():
        try:
            df = pd.read_csv(cache, index_col="datetime", parse_dates=True)
            print(f"  {YLW}SILVERMIC  {len(df):>6} bars  (stale cache: {df.index[-1]:%Y-%m-%d}){RST}")
            return df
        except Exception:
            pass

    raise RuntimeError("No warmup data available (API failed, no cache)")


# ─── Indicator Engine ─────────────────────────────────────────────────────


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
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
) -> tuple[str | None, str | None]:
    """Evaluate prearmed PDL trigger. Returns (direction, trigger_name)."""
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

    if prev_day_high is not None:
        long_trigger = prev_day_high + buffer
        if ("LONG" not in dirs
                and prev_close <= prev_day_high + tick_tol
                and h >= long_trigger
                and c >= prev_day_high):
            return "LONG", "PDL_BASE"

    if prev_day_low is not None:
        short_trigger = prev_day_low - buffer
        if ("SHORT" not in dirs
                and prev_close >= prev_day_low - tick_tol
                and l <= short_trigger
                and c <= prev_day_low):
            return "SHORT", "PDL_BASE"

    return None, None


# ─── Position Sizing & Risk Checks ───────────────────────────────────────


def check_trade_risk(atr: float) -> tuple[bool, str]:
    """Return (ok, reason) — reject if SL risk exceeds per-trade limit."""
    sl_risk = atr * PDL_SL_MULT * LOT_SIZE
    if sl_risk > MAX_SL_RISK_PER_TRADE:
        return False, f"sl_risk=₹{sl_risk:,.0f} > ₹{MAX_SL_RISK_PER_TRADE:,.0f}"
    return True, ""


def check_margin(price: float, capital: float) -> tuple[bool, str]:
    """Return (ok, reason) — reject if margin exceeds available capital."""
    margin_needed = price * LOT_SIZE * MARGIN_PCT
    if margin_needed > capital:
        return False, f"margin=₹{margin_needed:,.0f} > capital=₹{capital:,.0f}"
    return True, ""


# ─── Display ──────────────────────────────────────────────────────────────


def print_signal(direction, price, atr, sl, tp, ts=None):
    clr = GRN if direction == "LONG" else RED
    arrow = "▲" if direction == "LONG" else "▼"
    stamp = (ts or _now()).strftime("%H:%M:%S")
    sl_dist = atr * PDL_SL_MULT
    risk = sl_dist * LOT_SIZE
    print(f"\n{'=' * 60}")
    print(f"{clr}{BLD}  {arrow} {direction} [Prearmed PDL] — SILVERMIC  [{stamp}]{RST}")
    print(f"{'=' * 60}")
    print(f"  Price  : ₹{price:,.1f}")
    print(f"  SL     : ₹{sl:,.1f}  ({PDL_SL_MULT}× ATR = ₹{sl_dist:.1f})")
    print(f"  TP     : ₹{tp:,.1f}  (RR 1:{RR})")
    print(f"  Qty    : {LOT_SIZE} lot  (risk ₹{risk:,.0f})")
    print(f"{'=' * 60}\n")


def format_tg_signal(direction, price, atr, sl, tp):
    icon = "\U0001f7e2" if direction == "LONG" else "\U0001f534"
    sl_dist = atr * PDL_SL_MULT
    risk = sl_dist * LOT_SIZE
    return (
        f"{icon} *{direction} [Prearmed PDL] — SILVERMIC*\n"
        f"Price: ₹{price:,.1f}\n"
        f"SL: ₹{sl:,.1f} | TP: ₹{tp:,.1f}\n"
        f"Lot: {LOT_SIZE} | Risk: ₹{risk:,.0f}\n"
        f"RR: 1:{RR}"
    )


def print_dashboard(close_price: float, atr: float, pdl_h, pdl_l,
                     phase: str, virtual_pos=None, virtual_pnl: float = 0.0):
    now = _now()
    phase_colors = {
        "ACTIVE": f"{GRN}{BLD}Active — signals ON{RST}",
        "WIND_DOWN": f"{YLW}Wind-Down — no new entries{RST}",
        "WARMUP": f"{CYN}Warm-Up{RST}",
        "PRE_SESSION": f"{DIM}Pre-session{RST}",
        "SHUTDOWN": f"{DIM}Shutdown{RST}",
    }
    phase_str = phase_colors.get(phase, phase)

    print(f"\n{CYN}[{now:%H:%M:%S}]{RST} {phase_str}")
    print(f"  {'─' * 45}")
    print(f"  SILVERMIC  Close: ₹{close_price:>10,.1f}  ATR: ₹{atr:>7,.1f}")
    if pdl_h is not None:
        print(f"  PDL        High:  ₹{pdl_h:>10,.1f}  Low: ₹{pdl_l:>10,.1f}")
    if virtual_pos:
        d = virtual_pos["direction"]
        entry = virtual_pos["entry_price"]
        if d == "LONG":
            unreal = (close_price - entry) * LOT_SIZE
        else:
            unreal = (entry - close_price) * LOT_SIZE
        pnl_s = f"{GRN}+₹{unreal:,.0f}{RST}" if unreal >= 0 else f"{RED}-₹{abs(unreal):,.0f}{RST}"
        print(f"  Position   {d:>5} @ ₹{entry:,.1f}  unrealized {pnl_s}")
    print(f"  Day P&L    ₹{virtual_pnl:+,.0f}")


# ─── Signal Log ──────────────────────────────────────────────────────────

_LOG_COLUMNS = [
    "date", "time", "direction", "trigger", "price",
    "sl", "tp", "atr", "lot", "risk", "phase",
]


def log_signal(direction, trigger, price, sl, tp, atr, ts=None, phase=""):
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = ts or _now()
    log_date = stamp.strftime("%Y-%m-%d")
    log_file = SIGNALS_DIR / f"silvermic_{log_date}.csv"
    is_new = not log_file.exists()

    risk = abs(price - sl) * LOT_SIZE
    row = {
        "date": log_date,
        "time": stamp.strftime("%H:%M:%S"),
        "direction": direction,
        "trigger": trigger or "",
        "price": round(price, 1),
        "sl": round(sl, 1),
        "tp": round(tp, 1),
        "atr": round(atr, 1),
        "lot": LOT_SIZE,
        "risk": round(risk, 0),
        "phase": phase,
    }

    with open(log_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_LOG_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


# ─── State Persistence ────────────────────────────────────────────────────


def _save_state(session_date, cooldowns, pdl_dirs_used, daily_sl_count,
                virtual_pos, signal_count, today_signals, virtual_pnl):
    try:
        pos_ser = None
        if virtual_pos:
            pos_ser = {}
            for k, v in virtual_pos.items():
                if isinstance(v, (pd.Timestamp, datetime)):
                    pos_ser[k] = v.isoformat()
                else:
                    pos_ser[k] = v

        state = {
            "date": str(session_date),
            "cooldowns": {k: v.isoformat() for k, v in cooldowns.items()},
            "pdl_dirs_used": list(pdl_dirs_used),
            "daily_sl_count": daily_sl_count,
            "signal_count": signal_count,
            "virtual_pos": pos_ser,
            "virtual_pnl": virtual_pnl,
            "n_signals": len(today_signals),
        }
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(state, indent=2))
    except Exception:
        pass


def _load_state(session_date):
    if not STATE_FILE.exists():
        return None
    try:
        state = json.loads(STATE_FILE.read_text())
        if state.get("date") != str(session_date):
            return None
        cooldowns = {}
        for k, v in state.get("cooldowns", {}).items():
            cooldowns[k] = datetime.fromisoformat(v)
        return {
            "cooldowns": cooldowns,
            "pdl_dirs_used": set(state.get("pdl_dirs_used", [])),
            "daily_sl_count": state.get("daily_sl_count", 0),
            "signal_count": state.get("signal_count", 0),
            "virtual_pos": state.get("virtual_pos"),
            "virtual_pnl": float(state.get("virtual_pnl", 0.0)),
        }
    except Exception:
        return None


def _load_weekly_loss() -> float:
    """Load cumulative weekly virtual loss (resets every Monday)."""
    if not WEEKLY_FILE.exists():
        return 0.0
    try:
        data = json.loads(WEEKLY_FILE.read_text())
        week_start = data.get("week_start", "")
        today = _now().date()
        iso_year, iso_week, _ = today.isocalendar()
        expected_start = date.fromisocalendar(iso_year, iso_week, 1).isoformat()
        if week_start == expected_start:
            return float(data.get("loss", 0.0))
        return 0.0
    except Exception:
        return 0.0


def _save_weekly_loss(total_loss: float):
    try:
        today = _now().date()
        iso_year, iso_week, _ = today.isocalendar()
        week_start = date.fromisocalendar(iso_year, iso_week, 1).isoformat()
        WEEKLY_FILE.parent.mkdir(parents=True, exist_ok=True)
        WEEKLY_FILE.write_text(json.dumps({
            "week_start": week_start,
            "loss": round(total_loss, 2),
        }))
    except Exception:
        pass


# ─── Virtual Bracket Exit ─────────────────────────────────────────────────


def _check_bracket_hit(pos: dict, bar_high: float, bar_low: float) -> str | None:
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


# ─── Phases ───────────────────────────────────────────────────────────────


def get_phase(t: time) -> str:
    if t < MCX_OPEN:
        return "PRE_SESSION"
    if t < ENTRY_START:
        return "WARMUP"
    if t <= ENTRY_END:
        return "ACTIVE"
    bar_grace = (datetime.combine(datetime.min, ENTRY_END) + BAR_INTERVAL).time()
    if t <= bar_grace:
        return "ACTIVE"
    if t <= FORCE_EXIT_TIME:
        return "WIND_DOWN"
    return "SHUTDOWN"


def in_entry_window(t: time) -> bool:
    return ENTRY_START <= t <= ENTRY_END


# ─── Scan Logic ───────────────────────────────────────────────────────────


def scan_bar(df, bar_idx, prev_day_high, prev_day_low, pdl_dirs_used):
    """Check one bar for a prearmed PDL signal. Returns result dict."""
    row = df.iloc[bar_idx]
    prev_row = df.iloc[bar_idx - 1] if bar_idx != 0 and abs(bar_idx) < len(df) else None
    ts = df.index[bar_idx]

    sig, trigger = check_signal(row, prev_row, prev_day_high, prev_day_low, pdl_dirs_used)

    result = {
        "close": row["close"],
        "atr": row["atr"] if not pd.isna(row.get("atr", float("nan"))) else 0.0,
        "signal": sig,
        "trigger": trigger,
        "pdl_high": prev_day_high,
        "pdl_low": prev_day_low,
    }

    if sig:
        a = row["atr"]
        p = (
            prev_day_high + a * PDL_PREARM_BUFFER_ATR
            if sig == "LONG" else prev_day_low - a * PDL_PREARM_BUFFER_ATR
        )
        p = round(p / TICK_SIZE) * TICK_SIZE
        sl_dist = a * PDL_SL_MULT
        sl = round((p - sl_dist if sig == "LONG" else p + sl_dist) / TICK_SIZE) * TICK_SIZE
        tp = round((p + sl_dist * RR if sig == "LONG" else p - sl_dist * RR) / TICK_SIZE) * TICK_SIZE
        result.update(price=p, sl=sl, tp=tp, ts=ts, sl_mult=PDL_SL_MULT)

    return result


# ─── Fetch Latest ─────────────────────────────────────────────────────────


def _fetch_latest(client, token, stock_data, now):
    """Fetch candles since the last known bar and merge."""
    last_bar_ts = stock_data.index[-1] if not stock_data.empty else None
    if last_bar_ts:
        gap_mins = max(10, int((now - last_bar_ts).total_seconds() / 60) + 5)
        lookback = min(gap_mins, 120)
    else:
        lookback = 30

    window_start = (now - timedelta(minutes=lookback)).strftime("%Y-%m-%d %H:%M")
    window_end = (now - timedelta(minutes=3)).strftime("%Y-%m-%d %H:%M")

    fresh = fetch_mcx_candles(client, token, window_start, window_end, retries=2)

    base = now.replace(hour=9, minute=0, second=0, microsecond=0)
    elapsed = (now - base).total_seconds()
    if elapsed >= 0:
        current_candle_open = base + timedelta(seconds=int(elapsed / 180) * 180)
        fresh = fresh[fresh.index < current_candle_open]

    if fresh.empty:
        return stock_data

    combined = pd.concat([stock_data, fresh])
    return combined[~combined.index.duplicated(keep="last")].sort_index()


# ─── Dry Run ──────────────────────────────────────────────────────────────


def dry_run(capital: float) -> None:
    print(f"\n{CYN}[api]{RST} Connecting to Angel One...")
    client = AngelOneClient()
    client.connect()

    print(f"\n{CYN}[contract]{RST} Resolving front-month SILVERMIC...")
    tsym, token = resolve_front_month(client)
    print(f"  {GRN}{tsym}  token={token}{RST}")

    print(f"\n{CYN}[data]{RST} Loading warmup data...")
    data = load_warmup_cached(client, token)
    if data.empty:
        print(f"{RED}No data available.{RST}")
        return

    scan_date = data.index[-1].date()
    df = compute_indicators(data)
    pdl_h, pdl_l = get_prev_day_levels(df, scan_date)

    if pdl_h is None:
        print(f"{YLW}No prev-day data for {scan_date}{RST}")
        return

    print(f"\n{CYN}[pdl]{RST} Prev-Day  H=₹{pdl_h:,.1f}  L=₹{pdl_l:,.1f}")
    print(f"\n{YLW}{BLD}── DRY RUN: Scanning {scan_date} ──{RST}\n")

    signals_found = 0
    cooldowns: dict[str, datetime] = {}
    pdl_dirs_used: set[str] = set()
    daily_sl_count = 0
    virtual_pos: dict | None = None
    virtual_pnl = 0.0

    day_mask = df.index.date == scan_date
    for i in df.index[day_mask]:
        idx = df.index.get_loc(i)
        row = df.iloc[idx]
        bar_done = i + BAR_INTERVAL

        if virtual_pos is not None:
            hit = _check_bracket_hit(virtual_pos, row["high"], row["low"])
            if hit:
                exit_price = virtual_pos["sl"] if hit == "SL" else virtual_pos["tp"]
                if virtual_pos["direction"] == "LONG":
                    pnl = (exit_price - virtual_pos["entry_price"]) * LOT_SIZE
                else:
                    pnl = (virtual_pos["entry_price"] - exit_price) * LOT_SIZE
                virtual_pnl += pnl
                clr = GRN if pnl >= 0 else RED
                print(f"  {clr}[{bar_done:%H:%M}] {hit} — ₹{pnl:+,.0f}  (day ₹{virtual_pnl:+,.0f}){RST}")
                if hit == "SL":
                    daily_sl_count += 1
                virtual_pos = None

        if virtual_pos is not None:
            if bar_done.time() >= FORCE_EXIT_TIME:
                exit_price = row["close"]
                if virtual_pos["direction"] == "LONG":
                    pnl = (exit_price - virtual_pos["entry_price"]) * LOT_SIZE
                else:
                    pnl = (virtual_pos["entry_price"] - exit_price) * LOT_SIZE
                virtual_pnl += pnl
                clr = GRN if pnl >= 0 else RED
                print(f"  {clr}[{bar_done:%H:%M}] EOD EXIT — ₹{pnl:+,.0f}  (day ₹{virtual_pnl:+,.0f}){RST}")
                virtual_pos = None

        if virtual_pos is not None:
            continue
        if not in_entry_window(bar_done.time()):
            continue
        on_cooldown = "last" in cooldowns and bar_done < cooldowns["last"]
        if on_cooldown:
            continue
        if daily_sl_count >= MAX_SL_PER_DAY:
            continue
        if virtual_pnl <= -MAX_DAILY_LOSS:
            continue

        result = scan_bar(df, idx, pdl_h, pdl_l, pdl_dirs_used)
        if not result["signal"]:
            continue

        risk_ok, risk_reason = check_trade_risk(result["atr"])
        if not risk_ok:
            flog.info("SKIP  %s  %s", result["signal"], risk_reason)
            continue

        margin_ok, margin_reason = check_margin(result["price"], capital)
        if not margin_ok:
            flog.info("SKIP  %s  %s", result["signal"], margin_reason)
            continue

        signals_found += 1
        cooldowns["last"] = bar_done + COOLDOWN
        pdl_dirs_used.add(result["signal"])

        r = result
        print_signal(r["signal"], r["price"], r["atr"], r["sl"], r["tp"], ts=r["ts"])
        log_signal(r["signal"], r["trigger"], r["price"], r["sl"], r["tp"], r["atr"],
                   ts=r["ts"], phase="ACTIVE")

        virtual_pos = {
            "direction": r["signal"],
            "entry_price": r["price"],
            "sl": r["sl"],
            "tp": r["tp"],
        }

    log_file = SIGNALS_DIR / f"silvermic_{scan_date}.csv"
    print(f"\n{CYN}── Dry run complete: {signals_found} signal(s)  Virtual P&L: ₹{virtual_pnl:+,.0f} ──{RST}")
    if signals_found:
        print(f"{DIM}  Log saved: {log_file}{RST}")


# ─── Live Loop ────────────────────────────────────────────────────────────


def live(capital: float) -> None:
    now_t = _now().time()
    if now_t > FORCE_EXIT_TIME:
        print(f"\n{YLW}{BLD}Evening session is over (current time: {now_t:%H:%M}).{RST}")
        print(f"{YLW}Active window is {ENTRY_START:%H:%M}–{ENTRY_END:%H:%M}.{RST}")
        print(f"{YLW}Waiting for next session (warmup at 17:45)...{RST}\n")
        while True:
            now = _now()
            if now.weekday() < 5 and time(17, 30) <= now.time() < FORCE_EXIT_TIME:
                break
            _time.sleep(60)
        print(f"{GRN}{BLD}Pre-session detected — resuming startup.{RST}\n")

    _attach_log_file()

    flog.info("=" * 70)
    flog.info("SESSION START  Capital=%.0f  RR=1:%.1f  MaxSL/day=%d  MaxDailyLoss=%.0f",
              capital, RR, MAX_SL_PER_DAY, MAX_DAILY_LOSS)
    flog.info("Strategy: PDL (buffer=%.2f×ATR  SL=%.1f×ATR)  SILVERMIC 1-lot", PDL_PREARM_BUFFER_ATR, PDL_SL_MULT)

    # ── Phase 1: Warm-Up ──────────────────────────────────────────────
    print(f"\n{CYN}{BLD}Phase 1 · Warm-Up{RST}")
    print(f"{CYN}[api] Connecting to Angel One...{RST}")
    flog.info("PHASE 1  Warm-Up — connecting")
    client = AngelOneClient()
    client.connect()
    flog.info("API connected")

    print(f"\n{CYN}[contract] Resolving front-month SILVERMIC...{RST}")
    tsym, token = resolve_front_month(client)
    print(f"  {GRN}{tsym}  token={token}{RST}")
    flog.info("CONTRACT  %s  token=%s", tsym, token)

    print(f"\n{CYN}[data] Fetching {WARMUP_3MIN_DAYS}-day 3-min candles{RST}")
    stock_data = load_warmup_cached(client, token)

    if stock_data.empty:
        print(f"{RED}No data loaded. Exiting.{RST}")
        flog.error("No data loaded — exiting")
        return

    pdl_h, pdl_l = get_prev_day_levels(stock_data, _now().date())
    if pdl_h is not None:
        print(f"\n{CYN}[pdl]{RST} Prev-Day  H=₹{pdl_h:,.1f}  L=₹{pdl_l:,.1f}")
        flog.info("PDL  H=%.1f  L=%.1f", pdl_h, pdl_l)
    else:
        print(f"  {YLW}No prev-day levels available{RST}")
        flog.warning("PDL  no prev-day data")

    if TG_TOKEN and TG_CHAT:
        print(f"{GRN}[tg] Telegram notifications enabled{RST}")
    else:
        print(f"{YLW}[tg] Not configured — console only{RST}")

    weekly_loss = _load_weekly_loss()

    print(f"\n{BLD}{'═' * 60}")
    print(f"  SILVER MICRO SIGNAL MONITOR — PDL Breakout")
    print(f"  Capital   : ₹{capital:,.0f}  |  RR 1:{RR}")
    print(f"  Contract  : {tsym}")
    print(f"  Lot       : {LOT_SIZE} kg  |  Margin ~{MARGIN_PCT*100:.1f}%")
    print(f"  SL        : {PDL_SL_MULT}× ATR  |  Max/trade ₹{MAX_SL_RISK_PER_TRADE:,.0f}")
    print(f"  Max Loss  : ₹{MAX_DAILY_LOSS:,.0f}/day  |  ₹{MAX_WEEKLY_LOSS:,.0f}/week")
    print(f"  Weekly    : ₹{weekly_loss:+,.0f} so far")
    print(f"  Mode      : {'AUTO-TRADE' if ENABLE_AUTO_TRADE else 'OBSERVATION ONLY'}")
    print(f"  {'─' * 56}")
    print(f"  17:45  Phase 1  Warm-Up       ✓ done")
    print(f"  18:00  Phase 2  Active         signals ON")
    print(f"  22:30  Phase 3  Wind-Down      no new entries")
    print(f"  23:10  Phase 4  Shutdown       force exit")
    print(f"{'═' * 60}{RST}\n")

    send_telegram(
        f"\U0001f514 *Silver Micro Monitor Started*\n"
        f"Capital: ₹{capital:,.0f}\n"
        f"Contract: `{tsym}`\n"
        f"Mode: {'Auto-Trade' if ENABLE_AUTO_TRADE else 'Observation'}\n"
        f"Window: 18:00–22:30 | SL: {PDL_SL_MULT}× ATR | RR: 1:{RR}"
    )

    cooldowns: dict[str, datetime] = {}
    pdl_dirs_used: set[str] = set()
    daily_sl_count = 0
    signal_count = 0
    virtual_pos: dict | None = None
    virtual_pnl: float = 0.0
    today_signals: list[dict] = []
    last_processed_bar_ts: pd.Timestamp | None = (
        stock_data.index[-1] if not stock_data.empty else None
    )

    saved = _load_state(_now().date())
    if saved:
        cooldowns = saved["cooldowns"]
        pdl_dirs_used = saved["pdl_dirs_used"]
        daily_sl_count = saved["daily_sl_count"]
        signal_count = saved["signal_count"]
        virtual_pos = saved.get("virtual_pos")
        virtual_pnl = saved.get("virtual_pnl", 0.0)
        flog.info("STATE RESTORED  sl_count=%d  signals=%d  pnl=₹%.0f",
                  daily_sl_count, signal_count, virtual_pnl)
        print(f"  {GRN}[state] Restored session state{RST}")

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
                if virtual_pnl < 0:
                    weekly_loss += abs(virtual_pnl)
                    _save_weekly_loss(weekly_loss)
                current_date = now.date()
                cooldowns.clear()
                pdl_dirs_used.clear()
                daily_sl_count = 0
                signal_count = 0
                virtual_pos = None
                virtual_pnl = 0.0
                today_signals.clear()
                scan_count = 0
                last_scan_candle = -1
                last_phase = ""
                # Weekly reset
                if now.weekday() == 0:
                    weekly_loss = 0.0
                    _save_weekly_loss(0.0)
                    flog.info("WEEKLY RESET")
                pdl_h, pdl_l = get_prev_day_levels(stock_data, current_date)
                if pdl_h is not None:
                    print(f"\n{CYN}[new day] {current_date}  PDL H=₹{pdl_h:,.1f}  L=₹{pdl_l:,.1f}{RST}")
                    flog.info("PDL  H=%.1f  L=%.1f", pdl_h, pdl_l)

            # Phase transition
            if phase != last_phase:
                prev_phase = last_phase
                last_phase = phase
                flog.info("PHASE  %s → %s", prev_phase or "START", phase)

                if phase == "ACTIVE":
                    print(f"\n{GRN}{BLD}▶ Active — signals ENABLED{RST}")
                    send_telegram("\u25b6\ufe0f *Active* — Silver Micro signals enabled")

                elif phase == "WIND_DOWN":
                    print(f"\n{YLW}▶ Wind-Down — no new entries{RST}")

                elif phase == "SHUTDOWN":
                    n = len(today_signals)

                    if virtual_pos is not None:
                        exit_price = stock_data.iloc[-1]["close"] if not stock_data.empty else virtual_pos["entry_price"]
                        if virtual_pos["direction"] == "LONG":
                            pnl = (exit_price - virtual_pos["entry_price"]) * LOT_SIZE
                        else:
                            pnl = (virtual_pos["entry_price"] - exit_price) * LOT_SIZE
                        virtual_pnl += pnl
                        clr = GRN if pnl >= 0 else RED
                        print(f"  {clr}[{now:%H:%M}] FORCE EXIT — ₹{pnl:+,.0f}  (day ₹{virtual_pnl:+,.0f}){RST}")
                        flog.info("FORCE EXIT  ₹%+.0f  day_total=₹%+.0f", pnl, virtual_pnl)
                        virtual_pos = None

                    print(f"\n{DIM}[{now:%H:%M}] Shutdown. {n} signal(s) today. Virtual P&L: ₹{virtual_pnl:+,.0f}{RST}")
                    flog.info("=" * 70)
                    flog.info("SHUTDOWN  %d signal(s)  virtual_pnl=₹%+.0f", n, virtual_pnl)

                    if virtual_pnl < 0:
                        weekly_loss += abs(virtual_pnl)
                        _save_weekly_loss(weekly_loss)

                    for s in today_signals:
                        print(f"  {s['time']:%H:%M}  {s['dir']:>5}  @ ₹{s['price']:,.1f}")
                        flog.info("  SUMMARY  %s  %5s  @ %.1f  SL=%.1f  TP=%.1f",
                                  s["time"].strftime("%H:%M"), s["dir"], s["price"], s["sl"], s["tp"])

                    send_telegram(
                        f"\U0001f4f4 *Shutdown*  {n} signal(s)\n"
                        f"Virtual P&L: ₹{virtual_pnl:+,.0f}\n"
                        f"Weekly: ₹{-weekly_loss:+,.0f}"
                    )
                    break

            # Pre-session wait
            if phase == "PRE_SESSION":
                target = datetime.combine(now.date(), time(17, 30))
                wait = max(int((target - now).total_seconds()), 0)
                print(f"{DIM}[{now:%H:%M}] Pre-session. Evening start in ~{wait // 3600}h {wait % 3600 // 60}m...{RST}")
                _time.sleep(min(wait, 300) if wait > 5 else max(wait, 0.2))
                continue

            # Warmup wait (09:00-18:00 — track data silently)
            if phase == "WARMUP":
                target = datetime.combine(now.date(), ENTRY_START)
                wait = max(int((target - now).total_seconds()), 0)
                if wait > 300:
                    m, s = divmod(wait, 60)
                    h, m = divmod(m, 60)
                    print(f"{DIM}[{now:%H:%M}] Warming up. Active at {ENTRY_START:%H:%M} ({h}h {m}m)...{RST}")
                    _time.sleep(min(wait, 300))
                    continue
                elif wait > 0:
                    print(f"{DIM}[{now:%H:%M}] Active in {wait}s...{RST}")
                    _time.sleep(wait)
                    continue

            # Align to 3-min candle grid (base 09:00)
            now_ts = _now()
            _base = now_ts.replace(hour=9, minute=0, second=0, microsecond=0)
            _elapsed = (now_ts - _base).total_seconds()
            _candle_idx = int(_elapsed / 180) if _elapsed >= 0 else -1

            if _candle_idx <= last_scan_candle:
                nxt = _base + timedelta(seconds=(last_scan_candle + 1) * 180 + POLL_BUFFER_SEC)
                wait_s = max(0, (nxt - _now()).total_seconds())
                m, s = divmod(int(wait_s), 60)
                phase_lbl = "Active" if phase == "ACTIVE" else "Wind-Down"
                sys.stdout.write(
                    f"\r{DIM}[{now:%H:%M:%S}] {phase_lbl} — "
                    f"next scan {nxt:%H:%M:%S} ({m}m {s}s)   {RST}"
                )
                sys.stdout.flush()
                _time.sleep(min(max(wait_s, 0.5), 30))
                continue

            _poll_at = _base + timedelta(seconds=_candle_idx * 180 + POLL_BUFFER_SEC)
            scan_bar_completed_ts = pd.Timestamp(_base + timedelta(seconds=_candle_idx * 180))
            _wait_poll = (_poll_at - _now()).total_seconds()
            if _wait_poll > 0:
                _time.sleep(_wait_poll)

            missed = _candle_idx - last_scan_candle - 1
            if missed > 0:
                flog.warning("MISSED %d candle(s) (idx %d→%d)", missed, last_scan_candle, _candle_idx)
            last_scan_candle = _candle_idx

            # ── Scan cycle ────────────────────────────────────────────
            scan_count += 1
            scan_start = _time.time()
            flog.info("SCAN #%d  %s  phase=%s", scan_count, now.strftime("%H:%M:%S"), phase)

            try:
                stock_data = _fetch_latest(client, token, stock_data, now)
            except Exception as exc:
                flog.warning("FETCH failed: %s", exc)
                if "Invalid Token" in str(exc):
                    try:
                        tsym, token = resolve_front_month(client)
                        flog.info("TOKEN REFRESH  %s  token=%s", tsym, token)
                        print(f"  {YLW}[contract] Refreshed: {tsym}{RST}")
                    except Exception as re_exc:
                        flog.error("TOKEN REFRESH failed: %s", re_exc)

            df = compute_indicators(stock_data)
            if pdl_h is None:
                pdl_h, pdl_l = get_prev_day_levels(df, current_date)
                if pdl_h is not None:
                    print(f"  {CYN}[pdl] Resolved: H=₹{pdl_h:,.1f}  L=₹{pdl_l:,.1f}{RST}")
                    flog.info("PDL RESOLVED  H=%.1f  L=%.1f", pdl_h, pdl_l)

            last_seen = last_processed_bar_ts
            pending_ts = list(df.index[df.index > last_seen]) if last_seen is not None else list(df.index[-5:])

            for bar_ts in pending_ts:
                bar_idx = df.index.get_loc(bar_ts)
                bar_row = df.iloc[bar_idx]
                bar_completed_ts = bar_ts + BAR_INTERVAL
                bar_phase = get_phase(bar_completed_ts.time())

                # Virtual bracket check
                if virtual_pos is not None:
                    hit = _check_bracket_hit(virtual_pos, bar_row["high"], bar_row["low"])
                    if hit:
                        exit_price = virtual_pos["sl"] if hit == "SL" else virtual_pos["tp"]
                        if virtual_pos["direction"] == "LONG":
                            pnl = (exit_price - virtual_pos["entry_price"]) * LOT_SIZE
                        else:
                            pnl = (virtual_pos["entry_price"] - exit_price) * LOT_SIZE
                        virtual_pnl += pnl
                        if hit == "SL":
                            daily_sl_count += 1
                        clr = GRN if pnl >= 0 else RED
                        print(f"\n  {clr}[{bar_completed_ts:%H:%M}] {hit} — ₹{pnl:+,.0f}  (day ₹{virtual_pnl:+,.0f}){RST}")
                        flog.info("BRACKET  %s  ₹%+.0f  day=₹%+.0f", hit, pnl, virtual_pnl)
                        send_telegram(
                            f"{'🟢' if pnl >= 0 else '🔴'} *{hit}* — SILVERMIC\n"
                            f"P&L: ₹{pnl:+,.0f}\nDay: ₹{virtual_pnl:+,.0f}"
                        )
                        virtual_pos = None

                    elif bar_completed_ts.time() >= FORCE_EXIT_TIME:
                        exit_price = bar_row["close"]
                        if virtual_pos["direction"] == "LONG":
                            pnl = (exit_price - virtual_pos["entry_price"]) * LOT_SIZE
                        else:
                            pnl = (virtual_pos["entry_price"] - exit_price) * LOT_SIZE
                        virtual_pnl += pnl
                        print(f"\n  {YLW}[{bar_completed_ts:%H:%M}] EOD EXIT — ₹{pnl:+,.0f}  (day ₹{virtual_pnl:+,.0f}){RST}")
                        flog.info("EOD EXIT  ₹%+.0f  day=₹%+.0f", pnl, virtual_pnl)
                        virtual_pos = None

                if virtual_pos is not None:
                    continue

                if bar_phase != "ACTIVE":
                    continue

                if not in_entry_window(bar_completed_ts.time()):
                    continue

                on_cooldown = "last" in cooldowns and bar_completed_ts < cooldowns["last"]
                if on_cooldown:
                    continue
                if daily_sl_count >= MAX_SL_PER_DAY:
                    continue
                if signal_count >= MAX_SIGNALS_PER_SESSION:
                    continue
                if virtual_pnl <= -MAX_DAILY_LOSS:
                    continue
                if weekly_loss + max(0, -virtual_pnl) >= MAX_WEEKLY_LOSS:
                    continue

                result = scan_bar(df, bar_idx, pdl_h, pdl_l, pdl_dirs_used)
                if not result["signal"]:
                    continue

                risk_ok, risk_reason = check_trade_risk(result["atr"])
                if not risk_ok:
                    flog.info("SKIP  %s  %s", result["signal"], risk_reason)
                    continue

                margin_ok, margin_reason = check_margin(result["price"], capital + virtual_pnl)
                if not margin_ok:
                    flog.info("SKIP  %s  %s", result["signal"], margin_reason)
                    continue

                r = result
                print_signal(r["signal"], r["price"], r["atr"], r["sl"], r["tp"], ts=r["ts"])
                send_telegram(format_tg_signal(r["signal"], r["price"], r["atr"], r["sl"], r["tp"]))
                log_signal(r["signal"], r["trigger"], r["price"], r["sl"], r["tp"], r["atr"],
                           ts=r["ts"], phase=bar_phase)

                cooldowns["last"] = bar_completed_ts + COOLDOWN
                signal_count += 1
                pdl_dirs_used.add(r["signal"])

                virtual_pos = {
                    "direction": r["signal"],
                    "entry_price": r["price"],
                    "sl": r["sl"],
                    "tp": r["tp"],
                    "entry_time": bar_completed_ts.isoformat(),
                }

                today_signals.append({
                    "dir": r["signal"],
                    "price": r["price"],
                    "sl": r["sl"],
                    "tp": r["tp"],
                    "time": _now(),
                })

                flog.info("SIGNAL  %s  @ ₹%.1f  SL=%.1f  TP=%.1f  risk=₹%.0f",
                          r["signal"], r["price"], r["sl"], r["tp"],
                          abs(r["price"] - r["sl"]) * LOT_SIZE)

            if pending_ts:
                last_processed_bar_ts = pending_ts[-1]

            # Dashboard
            if not df.empty:
                last_row = df.iloc[-1]
                last_atr = last_row.get("atr", 0.0)
                if pd.isna(last_atr):
                    last_atr = 0.0
                print_dashboard(last_row["close"], last_atr, pdl_h, pdl_l,
                                phase, virtual_pos, virtual_pnl)

            scan_elapsed = _time.time() - scan_start
            flog.info("SCAN #%d complete in %.1fs", scan_count, scan_elapsed)

            _save_state(current_date, cooldowns, pdl_dirs_used, daily_sl_count,
                        virtual_pos, signal_count, today_signals, virtual_pnl)

    except KeyboardInterrupt:
        n = len(today_signals)
        flog.info("STOPPED MANUALLY  %d signal(s)  virtual_pnl=₹%+.0f", n, virtual_pnl)
        print(f"\n\n{YLW}[stopped] {n} signal(s) today  Virtual P&L: ₹{virtual_pnl:+,.0f}{RST}")
        for s in today_signals:
            print(f"  {s['time']:%H:%M}  {s['dir']:>5}  @ ₹{s['price']:,.1f}")
        if virtual_pnl < 0:
            weekly_loss += abs(virtual_pnl)
            _save_weekly_loss(weekly_loss)
        send_telegram(f"\u23f9 Monitor stopped. {n} signal(s). P&L: ₹{virtual_pnl:+,.0f}")


# ─── CLI Entry ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="MCX Silver Micro — Live PDL Breakout Signal Monitor",
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

    print(f"\n{BLD}MCX Silver Micro — PDL Breakout Signal Monitor{RST}")
    print(f"{DIM}Prearmed PDL ({PDL_PREARM_BUFFER_ATR:.1f}×ATR trigger, SL {PDL_SL_MULT}×ATR)")
    print(f"RR 1:{RR} | Max {MAX_SL_PER_DAY} SL/day | ₹{MAX_DAILY_LOSS:,.0f} daily cap{RST}")
    print(f"{DIM}Mode: {'AUTO-TRADE' if ENABLE_AUTO_TRADE else 'OBSERVATION ONLY'}{RST}\n")

    if args.dry_run:
        dry_run(args.capital)
    else:
        try:
            live(args.capital)
        except KeyboardInterrupt:
            flog.info("SESSION END  user interrupt")
            print(f"\n{YLW}Session ended by user.{RST}")
        except Exception as exc:
            flog.error("SESSION CRASH  %s: %s", type(exc).__name__, exc)
            print(f"\n{RED}CRASH: {exc}{RST}")
            raise


if __name__ == "__main__":
    main()

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
Strategy: EMA(9/21) crossover · VWAP · RSI(9) · Volume SMA(20) · ATR(14)
RR      : 1 : 1.5  (SL = 1×ATR, TP = 1.5×ATR)
Sizing  : 1% risk per trade, 5× leverage cap

Usage:
    python live_signals.py                  # default ₹50,000 capital
    python live_signals.py --capital 100000 # custom capital
    python live_signals.py --dry-run        # scan last session, no live loop
"""

import argparse
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


# ─── Strategy Parameters ──────────────────────────────────────────────────

FAST_EMA = 9
SLOW_EMA = 21
RSI_PERIOD = 9
VOL_SMA = 20
ATR_PERIOD = 14
HTF_EMA_PERIOD = 21
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
WARMUP_3MIN_DAYS = 2
REGIME_FETCH_DAYS = 60
POLL_BUFFER_SEC = 2

DATA_DIR = Path(__file__).parent / "data"

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
    if t < ENTRY_AM[1]:
        return "ACTIVE_AM"
    if t < ENTRY_PM[0]:
        return "MIDDAY"
    if t < ENTRY_PM[1]:
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


def check_signal(row, htf_val, use_htf: bool, regime_val=None) -> str | None:
    """Evaluate entry conditions on a single bar. Returns 'LONG', 'SHORT', or None."""
    c, vwap = row["close"], row["vwap"]
    rsi, vol, vsma, atr = row["rsi"], row["volume"], row["vol_sma"], row["atr"]

    if pd.isna(rsi) or pd.isna(vsma) or pd.isna(atr) or atr <= 0:
        return None

    allow_long = allow_short = True

    if regime_val is not None:
        allow_long = c > regime_val
        allow_short = c < regime_val

    if use_htf and htf_val is not None and not pd.isna(htf_val):
        allow_long = allow_long and c > htf_val
        allow_short = allow_short and c < htf_val

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
    return (ENTRY_AM[0] <= t < ENTRY_AM[1]) or (ENTRY_PM[0] <= t < ENTRY_PM[1])


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
                trend = f"{GRN}↑ Bullish{RST}" if last_close > regime[sym] else f"{RED}↓ Bearish{RST}"
                print(f"  {sym:<11} Daily EMA(21) ₹{regime[sym]:>9,.2f}  {trend}")
            else:
                print(f"  {YLW}{sym:<11} only {len(df)} daily bars (need {REGIME_EMA_PERIOD}){RST}")
            _time.sleep(0.3)
        except Exception as exc:
            print(f"  {RED}{sym:<11} daily fetch failed: {exc}{RST}")
    return regime


def load_warmup(client: AngelOneClient | None) -> dict[str, pd.DataFrame]:
    """Load 3-min warmup data: try cached CSV first, fall back to API."""
    data: dict[str, pd.DataFrame] = {}
    for sym, _ in STOCKS:
        cache = _cache_path(sym)
        if cache.exists():
            try:
                df = load_csv(str(cache))
                data[sym] = df
                print(f"  {sym:<11} {len(df):>6} bars  (cache: {df.index[-1]:%Y-%m-%d})")
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
            except Exception as exc:
                print(f"  {RED}{sym:<11} FAILED: {exc}{RST}")
            _time.sleep(0.3)
        else:
            print(f"  {YLW}{sym:<11} no data{RST}")
    return data


# ─── Signal Scan (shared by dry-run and live) ─────────────────────────────


def scan_bar(sym, sec, df, use_htf, htf_series, bar_idx, capital, regime_val=None):
    """Check one bar for a signal. Returns a result dict."""
    row = df.iloc[bar_idx]
    ts = df.index[bar_idx]
    htf_val = None
    if htf_series is not None and ts in htf_series.index:
        htf_val = htf_series.loc[ts]

    sig = check_signal(row, htf_val, use_htf, regime_val=regime_val)
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
    """Fetch the latest few candles and merge into existing data."""
    window_start = (now - timedelta(minutes=6)).strftime("%Y-%m-%d %H:%M")
    window_end = now.strftime("%Y-%m-%d %H:%M")
    fresh = client.fetch_candles(sym, window_start, window_end)
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

    ref_sym = next(iter(data))
    scan_date = data[ref_sym].index[-1].date()
    print(f"\n{YLW}{BLD}── DRY RUN: Scanning {scan_date} for signals ──{RST}\n")

    signals_found = 0
    cooldowns: dict[str, datetime] = {}
    for sym, sec in STOCKS:
        if sym not in data:
            continue
        df = compute_indicators(data[sym])
        use_htf = sym in HTF_STOCKS
        htf_s = compute_htf_ema(data[sym]) if use_htf else None
        regime_val = regime_emas.get(sym)

        day_mask = df.index.date == scan_date
        for i in df.index[day_mask]:
            if not in_entry_window(i.time()):
                continue
            on_cooldown = sym in cooldowns and i < cooldowns[sym]
            if on_cooldown:
                continue
            idx = df.index.get_loc(i)
            result = scan_bar(sym, sec, df, use_htf, htf_s, idx, capital, regime_val)
            if result["signal"]:
                signals_found += 1
                cooldowns[sym] = i + COOLDOWN
                r = result
                print_signal(
                    sym, sec, r["signal"], r["price"], r["atr"], r["qty"],
                    r["sl"], r["tp"], r["rsi"], r["vwap"], r["vol_ratio"],
                    r["htf_val"], use_htf, regime_val=regime_val, ts=r["ts"],
                )

    print(f"\n{CYN}── Dry run complete: {signals_found} signal(s) found on {scan_date} ──{RST}")


# ─── Live Loop (Phase-Based Architecture) ─────────────────────────────────


def live(capital: float) -> None:
    if TG_TOKEN and TG_CHAT:
        print(f"{GRN}[tg] Telegram notifications enabled{RST}")
    else:
        print(f"{YLW}[tg] Not configured — console only{RST}")
        print(_TELEGRAM_SETUP)

    # ── Phase 1: Warm-Up ──────────────────────────────────────────────
    print(f"\n{CYN}{BLD}Phase 1 · Warm-Up{RST}")
    print(f"{CYN}[api] Connecting to Angel One...{RST}")
    client = AngelOneClient()
    client.connect()

    print(f"\n{CYN}[regime] Fetching daily candles → Daily 21-EMA regime filter{RST}")
    regime_emas = fetch_regime_emas(client)

    print(f"\n{CYN}[data] Fetching {WARMUP_3MIN_DAYS}-day 3-min candles (indicator warmup){RST}")
    stock_data = load_warmup(client)
    loaded = len(stock_data)

    if loaded == 0:
        print(f"{RED}No data loaded. Exiting.{RST}")
        return

    syms = ", ".join(s for s, _ in STOCKS if s in stock_data)
    regime_count = len(regime_emas)
    htf_count = sum(1 for s, _ in STOCKS if s in HTF_STOCKS and s in stock_data)

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
    today_signals: list[dict] = []
    current_date = datetime.now().date()
    last_phase = ""

    try:
        while True:
            now = datetime.now()
            t = now.time()
            phase = get_phase(t)

            # Day rollover
            if now.date() != current_date:
                current_date = now.date()
                cooldowns.clear()
                today_signals.clear()
                last_phase = ""
                print(f"\n{CYN}[new day] {current_date}{RST}")
                print(f"{CYN}[regime] Refreshing daily EMAs...{RST}")
                try:
                    regime_emas = fetch_regime_emas(client)
                except Exception:
                    print(f"{YLW}  regime refresh failed — using yesterday's values{RST}")

            # Phase transition announcement
            if phase != last_phase:
                last_phase = phase
                lbl = PHASES.get(phase, phase)
                if phase in ("ACTIVE_AM", "ACTIVE_PM"):
                    print(f"\n{GRN}{BLD}▶ {lbl} — signals ENABLED{RST}")
                    send_telegram(f"\u25b6\ufe0f *{lbl}* — signals enabled")
                elif phase == "SHUTDOWN":
                    n = len(today_signals)
                    print(f"\n{DIM}[{now:%H:%M}] {lbl}. {n} signal(s) today.{RST}")
                    if today_signals:
                        for s in today_signals:
                            print(f"  {s['time']:%H:%M}  {s['dir']:>5}  {s['sym']:<11}  @ ₹{s['price']:,.2f}")
                    send_telegram(f"\U0001f4f4 Shutdown. {n} signal(s) today.")
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

            # Align to next HH:MM:02 candle boundary
            nxt = next_candle_time()
            wait_s = (nxt - datetime.now()).total_seconds()
            if wait_s > 1:
                lbl = PHASES.get(phase, "")
                m, s = divmod(int(wait_s), 60)
                sys.stdout.write(
                    f"\r{DIM}[{now:%H:%M:%S}] {lbl} — "
                    f"next scan {nxt:%H:%M:%S} ({m}m {s}s)   {RST}"
                )
                sys.stdout.flush()
                _time.sleep(min(wait_s, 30))
                continue

            # ── Scan cycle ────────────────────────────────────────────
            is_active = phase in ("ACTIVE_AM", "ACTIVE_PM")
            dashboard: list[dict] = []

            for sym, sec in STOCKS:
                if sym not in stock_data:
                    continue

                try:
                    _fetch_latest(client, sym, stock_data, now)
                except Exception as exc:
                    if "invalid" in str(exc).lower() or "session" in str(exc).lower():
                        try:
                            client.connect()
                            _fetch_latest(client, sym, stock_data, now)
                        except Exception:
                            print(f"  {RED}{sym}: reconnect failed{RST}")
                            continue
                    else:
                        print(f"  {DIM}{sym}: {exc}{RST}")
                        continue

                _time.sleep(0.1)

                df = compute_indicators(stock_data[sym])
                use_htf = sym in HTF_STOCKS
                htf_s = compute_htf_ema(stock_data[sym]) if use_htf else None
                regime_val = regime_emas.get(sym)

                result = scan_bar(
                    sym, sec, df, use_htf, htf_s, -1, capital, regime_val
                )

                on_cooldown = sym in cooldowns and now < cooldowns[sym]

                if is_active and result["signal"] and not on_cooldown:
                    r = result
                    print_signal(
                        sym, sec, r["signal"], r["price"], r["atr"], r["qty"],
                        r["sl"], r["tp"], r["rsi"], r["vwap"], r["vol_ratio"],
                        r["htf_val"], use_htf, regime_val=regime_val,
                    )
                    send_telegram(format_tg_signal(
                        sym, sec, r["signal"], r["price"], r["atr"], r["qty"],
                        r["sl"], r["tp"], r["rsi"], r["vwap"], r["vol_ratio"],
                    ))
                    cooldowns[sym] = now + COOLDOWN
                    today_signals.append({
                        "sym": sym, "dir": r["signal"],
                        "price": r["price"], "sl": r["sl"], "tp": r["tp"],
                        "time": now,
                    })
                else:
                    result["signal"] = None

                dashboard.append(result)

            print_dashboard(dashboard, phase)

    except KeyboardInterrupt:
        n = len(today_signals)
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

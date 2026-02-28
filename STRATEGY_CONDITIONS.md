# Signal Generation Conditions

Complete reference for all conditions evaluated to generate LONG/SHORT signals in `live_signals.py`.

---

## 1. Strategy Overview

**Hybrid Intraday Scalping Strategy** on NSE 3-minute candles.

Two entry types:
- **Entry A — PDL (Prev-Day Level Breakout)**: Price crosses yesterday's high/low with VWAP + RSI + Volume confirmation
- **Entry B — MOM (Momentum Breakout)**: Strong-body candle with sustained volume spike

Four hybrid enhancements over the baseline:

| # | Feature | What It Does |
|---|---------|--------------|
| 1 | **2-Bar Volume Confirmation** | Requires volume spike sustained over 2 consecutive bars (filters single-bar fake-outs) |
| 2 | **Dynamic Body Ratio** | 0.72 if ATR > 1.1× average, else 0.70 (demands stronger conviction in volatile conditions) |
| 3 | **ATR-Based SL Adjustment** | Widens SL 1.2× in high volatility, tightens 0.9× in low volatility |
| 4 | **Morning-Only Trading** | 10:00–12:00 only (avoids afternoon algorithmic chop) |

---

## 2. Stock Universe

Top 10 stocks selected by 90-day backtest P&L ranking across all Nifty 50:

| Stock | Sector |
|-------|--------|
| BAJAJ-AUTO | Auto |
| POWERGRID | Power |
| TATACONSUM | Consumer |
| INDIGO | Aviation |
| BAJFINANCE | Finance |
| HCLTECH | IT |
| JIOFIN | Finance |
| EICHERMOT | Auto |
| NESTLEIND | Consumer |
| BHARTIARTL | Telecom |

Re-evaluate periodically using `backtest_nifty50_90d.py`.

---

## 3. Indicators

All indicators computed on 3-minute OHLCV candles using pandas/numpy.

| Indicator | Formula | Purpose |
|-----------|---------|---------|
| **Session VWAP** | `Σ(TP × Vol) / Σ(Vol)`, resets daily. TP = (H+L+C)/3 | Intraday fair value anchor |
| **RSI(9)** | Wilder's smoothing: `ewm(alpha=1/9, adjust=False)` | Momentum filter |
| **Volume SMA(20)** | `volume.rolling(20).mean()` | Volume baseline |
| **ATR(14)** | TR = `max(H, PrevClose) - min(L, PrevClose)`, Wilder's: `ewm(alpha=1/14)` | Volatility for SL/TP sizing |
| **Body Ratio** | `abs(close - open) / (high - low)` | Candle conviction strength |
| **Volume Ratio** | `volume / vol_sma` | Relative volume spike detection |
| **ATR 20-avg** | `atr.rolling(20).mean()` | Volatility regime classification |

---

## 4. Entry Conditions

**All conditions in a column must be TRUE simultaneously for a signal to fire.**

### Entry A — PDL (Prev-Day Level Breakout)

| # | Condition | LONG | SHORT |
|---|-----------|------|-------|
| 1 | **Prev-Day Crossover** | prev_close ≤ prev_day_high AND close > prev_day_high | prev_close ≥ prev_day_low AND close < prev_day_low |
| 2 | **VWAP** | close > VWAP | close < VWAP |
| 3 | **RSI** | RSI > 50 | RSI < 50 |
| 4 | **Volume** | volume > vol_sma(20) | volume > vol_sma(20) |
| 5 | **Direction Limit** | Max 1 LONG PDL per stock per day | Max 1 SHORT PDL per stock per day |
| 6 | **Entry Window** | 10:00–12:00 IST | 10:00–12:00 IST |
| 7 | **Cooldown** | 15 min since last signal for this stock | 15 min since last signal for this stock |
| 8 | **Daily SL Guard** | < 1 SL hit today for this stock | < 1 SL hit today for this stock |

### Entry B — MOM (Momentum Breakout) — Hybrid

| # | Condition | LONG | SHORT |
|---|-----------|------|-------|
| 1 | **2-Bar Volume** | Both current and previous bar vol_ratio ≥ 1.6× (2.0 × 0.8) | Same |
| 2 | **Body Ratio** | body_ratio ≥ 0.72 (if ATR > 1.1× avg) or ≥ 0.70 (normal) | Same |
| 3 | **Direction** | close > open (bullish candle) | close < open (bearish candle) |
| 4 | **VWAP** | close > VWAP | close < VWAP |
| 5 | **RSI Range** | 30 ≤ RSI ≤ 85 | 15 ≤ RSI ≤ 70 |
| 6 | **Entry Window** | 10:00–12:00 IST | 10:00–12:00 IST |
| 7 | **Cooldown** | 15 min since last signal | 15 min since last signal |
| 8 | **Daily SL Guard** | < 1 SL hit today for this stock | < 1 SL hit today for this stock |

Fallback: If no previous bar data is available, a single-bar check with vol_ratio ≥ 2.0× is used instead of the 2-bar confirmation.

---

## 5. Time Windows

| Window | Time (IST) | Signals |
|--------|------------|---------|
| Pre-market | Before 09:15 | Blocked |
| Silent tracking | 09:15–09:59 | Blocked (indicators warming up) |
| **Morning window** | **10:00–12:00** | **Enabled** |
| Post-12:00 | 12:00–15:30 | Blocked (afternoon chop avoided) |
| Post-market | After 15:30 | Shutdown |

---

## 6. Risk Management

| Parameter | Value | Formula |
|-----------|-------|---------|
| **Risk per trade** | 1% of capital | `RISK_PCT = 0.01` |
| **PDL Stop Loss** | 1.5 × ATR (dynamic) | `entry ∓ ATR × 1.5 × sl_adj` |
| **MOM Stop Loss** | 1.2 × ATR (dynamic) | `entry ∓ ATR × 1.2 × sl_adj` |
| **Take Profit** | SL distance × 1.75 | RR = 1:1.75 |
| **Position Size** | `min(Capital × 1% / SL_dist, Capital × 5 / Price)` | Risk-based, capped by leverage |
| **Leverage Cap** | 5× (MIS intraday margin) | `LEV_CAP = 5.0` |
| **Cooldown** | 15 minutes per stock | Prevents rapid-fire signals |
| **Max SL per day** | 1 per stock | Blocks further entries after 1 SL hit |

### ATR-Based SL Adjustment (Hybrid)

The SL multiplier is dynamically scaled based on the volatility regime:

| Volatility Regime | Condition | SL Adjustment |
|--------------------|-----------|---------------|
| **High Volatility** | ATR / ATR_20avg > 1.3 | SL × 1.2 (wider, gives breathing room) |
| **Normal** | 0.7 ≤ ratio ≤ 1.3 | SL × 1.0 (no change) |
| **Low Volatility** | ATR / ATR_20avg < 0.7 | SL × 0.9 (tighter, less risk) |

---

## 7. Stagnation Exit

Tracks open positions and alerts to close if momentum stalls.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Trigger** | 8 bars (24 minutes) since entry | Enough time for 3-min momentum to establish |
| **Profit Gate** | Unrealized profit < 0.2 × ATR | If move hasn't cleared 0.2× ATR, it's a fakeout |
| **Action** | Console + Telegram EXIT alert | User cancels bracket order and closes at market |

---

## 8. Signal Flow Diagram

```
Every 3 minutes (at HH:MM:01)
│
├── Fetch latest 3-min candle for all 10 stocks
├── Recompute indicators (VWAP, RSI, Vol SMA, ATR, Body Ratio, Vol Ratio)
│
└── For each stock:
    │
    ├── Has open position?
    │   ├── Did bar hit SL or TP? ──────────── Yes ─→ Position closed (bracket filled)
    │   │                                              If SL → increment daily SL counter
    │   └── Stagnation check (8+ bars, < 0.2× ATR) ─→ ⚠ EXIT alert
    │
    ├── Is current time in 10:00–12:00? ──── No ──→ Skip
    ├── Is stock on cooldown? ──────────────  Yes ─→ Skip
    ├── Already in position? ──────────────  Yes ─→ Skip
    ├── Daily SL limit hit (≥1)? ──────────  Yes ─→ Skip
    │
    ├── Entry A: PDL Breakout?
    │   ├── Close crossed prev-day high/low (vs prev bar's close)
    │   ├── VWAP confirmation (above for LONG, below for SHORT)
    │   ├── RSI > 50 (LONG) or RSI < 50 (SHORT)
    │   ├── Volume > 20-period SMA
    │   ├── Direction not already used today (max 1 LONG + 1 SHORT per PDL)
    │   └── All true? → Signal fires (SL = 1.5× ATR, dynamic)
    │
    └── Entry B: MOM Breakout? (only if PDL didn't fire)
        ├── 2-bar volume: prev bar ≥ 1.6× AND current bar ≥ 1.6×
        ├── Body ratio ≥ 0.72 (high vol) or ≥ 0.70 (normal)
        ├── Bullish/bearish candle direction
        ├── VWAP confirmation
        ├── RSI in range: 30–85 (LONG) or 15–70 (SHORT)
        └── All true? → Signal fires (SL = 1.2× ATR, dynamic)
```

---

## 9. Daily Lifecycle

| Phase | Time | Action |
|-------|------|--------|
| **Phase 1 — Warm-Up** | 09:14 | Connect to Angel One, fetch 5 days of 3-min data, compute prev-day levels |
| **Phase 2 — Silent** | 09:15–09:59 | Track indicators every 3 min, NO signals (let indicators stabilize) |
| **Phase 3 — Active AM** | 10:00–12:00 | Signals enabled, dashboard displayed, Telegram alerts sent |
| **Phase 4 — Shutdown** | 12:00+ | Stop signals, continue tracking positions until SL/TP/stagnation |

---

## 10. Strategy Constants

```
RR             = 1.75       # Reward-to-risk ratio
RISK_PCT       = 0.01       # 1% risk per trade
LEV_CAP        = 5.0        # 5× MIS leverage cap

PDL_SL_MULT    = 1.5        # PDL entry: SL = 1.5× ATR
MOM_SL_MULT    = 1.2        # MOM entry: SL = 1.2× ATR
MOM_VOL_MULT   = 2.0        # MOM volume threshold (2× avg)
MOM_BODY_RATIO = 0.70       # Base body ratio (0.72 in high vol)

MAX_SL_PER_DAY = 1          # Max SL hits per stock before blocking
STAG_MAX_BARS  = 8          # Stagnation exit after 8 bars (24 min)
STAG_MIN_PROFIT_ATR = 0.2   # Min profit to avoid stagnation exit

MOM_LONG_RSI   = (30, 85)   # RSI range for MOM LONG
MOM_SHORT_RSI  = (15, 70)   # RSI range for MOM SHORT
PDL_LONG_RSI_MIN  = 50      # Min RSI for PDL LONG
PDL_SHORT_RSI_MAX = 50      # Max RSI for PDL SHORT

ENTRY_AM       = (10:00, 12:00)  # Morning trading window
ENTRY_PM       = None            # Afternoon disabled (hybrid)
COOLDOWN       = 15 minutes      # Per-stock cooldown
```

---

## 11. File Structure

```
trading/
├── live_signals.py          # Live signal generator (core)
├── fetch_data.py            # Angel One API data fetcher + CSV cache
├── backtest_nifty50_90d.py  # 90-day backtest on all Nifty 50, picks top 10
├── backtest_top10_90d.py    # 90-day backtest on current top 10 stocks
├── backtest_top10_5d.py     # Last 5 trading days backtest with trade log
├── data/                    # Cached 3-min OHLCV CSVs
└── logs/
    ├── live_signals/        # Daily live logs (live_YYYY-MM-DD.log)
    └── signals/             # Daily signal CSVs (signals_YYYY-MM-DD.csv)
```

---

## 12. Logging

| Output | Location | Content |
|--------|----------|---------|
| **Live logs** | `logs/live_signals/live_YYYY-MM-DD.log` | Timestamped events: warmup, signals, exits, errors |
| **Signal CSV** | `logs/signals/signals_YYYY-MM-DD.csv` | One row per signal: symbol, direction, trigger, price, SL, TP, ATR, qty, RSI, VWAP, vol_ratio |
| **Console** | stdout | Color-coded dashboard + signal alerts |
| **Telegram** | Bot API | Signal and exit alerts (if configured via .env) |

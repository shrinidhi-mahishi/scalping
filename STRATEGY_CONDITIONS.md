# Signal Generation Conditions

Complete reference for all conditions evaluated to generate BUY/SHORT signals across the screening and live signal scripts.

---

## 1. Stock Screening (`screen_nifty50.py`)

Screens all 48 Nifty 50 stocks to find the best candidates for live trading.

### Process

1. Fetch 90 days (configurable) of 3-minute OHLCV data for each stock
2. Run the full backtest strategy (`strategy.py`) in two modes per stock:
   - **Base** — no higher-timeframe filter
   - **HTF** — with 15-min 21-EMA trend confirmation enabled
3. Compare Base vs HTF results and pick the more profitable config
4. Rank all stocks by Net P&L

### Selection Criteria

| Metric | TRADEABLE | MARGINAL | AVOID |
|--------|-----------|----------|-------|
| Profit Factor | >= 1.0 | >= 0.7 | < 0.7 |
| Win Rate | >= 45% | any | any |

Only stocks marked **TRADEABLE** make the shortlist.

### Screener Output Per Stock

- Trades, Won, Lost, Win Rate %
- Net P&L (₹)
- Avg Win / Avg Loss
- Profit Factor
- Max Drawdown (% and ₹)
- Sharpe Ratio
- Long vs Short breakdown (count, wins, P&L)
- Best Side (LONG or SHORT)
- Recommended config (BASE or HTF)

---

## 2. Live Signal Generation (`live_signals.py`)

Polls Angel One every 3 minutes and evaluates all conditions in real-time on the shortlisted stocks.

### 2.1 Indicators

All indicators are computed on 3-minute OHLCV candles using pure pandas/numpy.

| Indicator | Formula | Purpose |
|-----------|---------|---------|
| **Session VWAP** | `Σ(TP × Vol) / Σ(Vol)`, resets daily. TP = (H+L+C)/3 | Intraday fair value anchor |
| **Fast EMA(9)** | `close.ewm(span=9, adjust=False)` | Short-term trend |
| **Slow EMA(21)** | `close.ewm(span=21, adjust=False)` | Medium-term trend |
| **EMA Crossover** | Fast EMA crosses above/below Slow EMA | Entry trigger |
| **RSI(9)** | Wilder's smoothing: `ewm(alpha=1/9, adjust=False)` | Momentum filter |
| **Volume SMA(20)** | `volume.rolling(20).mean()` | Volume baseline |
| **ATR(14)** | TR = `max(High, PrevClose) - min(Low, PrevClose)`, Wilder's smoothing: `ewm(alpha=1/14, adjust=False)` | Volatility for SL/TP |

### 2.2 Higher-Timeframe Filters

| Filter | Period | Data Source | Purpose |
|--------|--------|-------------|---------|
| **15-min 21-EMA (HTF)** | 21 periods on 15-min bars | 3-min candles resampled with `closed='right', label='right'` to avoid lookahead bias, then `reindex().ffill()` | Trend confirmation (per-stock, optional) |
| **Daily 21-EMA (Regime)** | 21 periods on daily bars | Daily candles fetched separately from API (60-day lookback) | Directional gate for all stocks |

### 2.3 Entry Conditions

**All conditions in a column must be TRUE simultaneously for a signal to fire.**

| # | Condition | LONG (BUY) | SHORT (SELL) |
|---|-----------|------------|--------------|
| 1 | **Regime Filter** | Close > Daily 21-EMA | Close < Daily 21-EMA |
| 2 | **HTF Filter** (if enabled for stock) | Close > 15-min 21-EMA | Close < 15-min 21-EMA |
| 3 | **VWAP** | Close > Session VWAP | Close < Session VWAP |
| 4 | **EMA Crossover** | EMA(9) crosses above EMA(21) | EMA(9) crosses below EMA(21) |
| 5 | **Volume** | Volume > 20-period SMA | Volume > 20-period SMA |
| 6 | **RSI Range** | 40 ≤ RSI(9) ≤ 70 | 30 ≤ RSI(9) ≤ 60 |
| 7 | **ATR Valid** | ATR > 0 and not NaN | ATR > 0 and not NaN |
| 8 | **Entry Window** | Within allowed time window | Within allowed time window |
| 9 | **Cooldown** | No signal for this stock in last 15 min | No signal for this stock in last 15 min |

### 2.4 Time Windows

| Window | Time (IST) | Signals |
|--------|------------|---------|
| Pre-market | Before 09:15 | Blocked |
| Silent tracking | 09:15 – 09:59 | Blocked (indicators updating) |
| **Morning window** | **10:00 – 12:00** | **Enabled** |
| Midday sleep | 12:00 – 13:59 | Blocked (indicators updating) |
| **Afternoon window** | **14:00 – 15:00** | **Enabled** |
| Post-market | After 15:00 | Shutdown |

### 2.5 Risk Management

| Parameter | Value | Formula |
|-----------|-------|---------|
| **Risk per trade** | 1% of capital | — |
| **Stop Loss** | 1 × ATR below/above entry | LONG: `Entry - ATR`, SHORT: `Entry + ATR` |
| **Take Profit** | 1.5 × ATR from entry | LONG: `Entry + 1.5×ATR`, SHORT: `Entry - 1.5×ATR` |
| **Risk-Reward Ratio** | 1 : 1.5 | — |
| **Position Size** | `min(Capital × 1% / ATR, Capital × 5 / Price)` | Capped by 5x MIS leverage |
| **Leverage Cap** | 5x (MIS intraday margin) | — |
| **Cooldown** | 15 minutes per stock after each signal | Prevents rapid-fire alerts |

### 2.6 Stagnation Exit

Tracks open positions after each entry signal and alerts to close if momentum fails.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Trigger** | 8 bars (24 minutes) since entry | Enough time for 3-min momentum to establish |
| **Profit Gate** | Unrealized profit < 0.2 × ATR | If you haven't cleared 0.2× ATR, the move was a fakeout |
| **Exemption** | HDFCLIFE | Slower-trending stock — stagnation exit harms its natural rhythm |
| **Action** | Console + Telegram EXIT alert | User manually cancels bracket order and closes at market |

How it works:
- After a LONG/SHORT signal fires, the script tracks entry price, ATR, and direction
- Every 3-min scan checks: has the bracket order filled (SL/TP hit)? If yes, position silently removed
- If position is still open after 8+ bars and profit < 0.2×ATR → stagnation exit alert fires
- Does **not** apply to HDFCLIFE (backtested as harmful for this stock)

Backtest result: +22.8% improvement over baseline across the shortlist.

### 2.7 Signal Flow Diagram

```
Every 3 minutes (at HH:MM:02)
│
├── Fetch latest 3-min candle for all stocks
├── Recompute indicators (VWAP, EMA, RSI, Vol SMA, ATR)
├── Recompute 15-min HTF EMA (for HTF stocks)
│
└── For each stock:
    │
    ├── Has open position?
    │   ├── Did bar hit SL or TP? ──────────── Yes ─→ Position closed (bracket filled)
    │   └── Stagnation check (8+ bars, < 0.2× ATR) ─→ ⚠ EXIT alert
    │
    ├── Is current time in entry window? ─── No ──→ Skip
    ├── Is stock on cooldown? ─────────────── Yes ─→ Skip
    ├── Already in position? ─────────────── Yes ─→ Skip
    │
    ├── Regime check: Close vs Daily 21-EMA
    │   ├── Close > Daily EMA → Only LONG allowed
    │   └── Close < Daily EMA → Only SHORT allowed
    │
    ├── HTF check (if enabled): Close vs 15-min 21-EMA
    │   ├── Close > HTF EMA → LONG confirmed
    │   └── Close < HTF EMA → SHORT confirmed
    │
    ├── LONG candidate?
    │   ├── Close > VWAP
    │   ├── EMA(9) just crossed above EMA(21)
    │   ├── Volume > Volume SMA(20)
    │   ├── 40 ≤ RSI(9) ≤ 70
    │   └── All true? → ▲ LONG SIGNAL (track position)
    │
    └── SHORT candidate?
        ├── Close < VWAP
        ├── EMA(9) just crossed below EMA(21)
        ├── Volume > Volume SMA(20)
        ├── 30 ≤ RSI(9) ≤ 60
        └── All true? → ▼ SHORT SIGNAL (track position)
```

### 2.8 Backtest vs Live Parity

The live signal script replicates the exact same indicator math and entry logic as `strategy.py` (used by the screener). Key implementation details ensuring parity:

| Aspect | Backtest (`strategy.py`) | Live (`live_signals.py`) |
|--------|--------------------------|--------------------------|
| VWAP | `SessionVWAP` indicator, resets per session | `groupby(date).cumsum()` on TP×Vol / Vol |
| EMA(9/21) | `bt.indicators.EMA(span=...)` | `ewm(span=..., adjust=False)` |
| RSI(9) | `bt.indicators.RSI(period=9)` | Wilder's: `ewm(alpha=1/9, adjust=False)` |
| ATR(14) | `bt.indicators.ATR(period=14)` | `max(H,PC)-min(L,PC)`, Wilder's: `ewm(alpha=1/14)` |
| EMA Cross | `bt.indicators.CrossOver` | `above & ~prev_above` / `~above & prev_above` |
| HTF EMA | `bt.indicators.EMA` on resampled 15-min data | `resample('15min', closed='right', label='right')` + `ffill` |
| Regime | Running daily EMA from end-of-day closes | Daily candles from API → `ewm(span=21)` |
| Position Sizing | `min(Capital×1%/ATR, Capital×5/Price)` | Same formula |
| Commission | `min(Value×0.03%, ₹20)` per leg | Not applied (signal-only, no execution) |
| Slippage | 0.01% on market/stop orders | Not applied (signal-only) |

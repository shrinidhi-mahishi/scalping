# NSE Intraday Scalping System

Automated stock screening and live signal generation for NSE intraday scalping.

**Strategy**: EMA(9/21) crossover + VWAP + RSI(9) + Volume SMA(20) + ATR(14) bracket  
**Filters**: Daily 21-EMA regime gate, optional 15-min 21-EMA HTF per stock  
**Risk**: 1% per trade, 1:1.5 RR, 5x MIS leverage cap

---

## Setup

```bash
cd trading
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your Angel One and Telegram credentials
```

### Required credentials (`.env`)

```
ANGEL_API_KEY=...          # Angel One SmartAPI key
ANGEL_CLIENT_ID=...        # Angel One client ID
ANGEL_PASSWORD=...         # Angel One PIN
ANGEL_TOTP_SECRET=...      # Base32 TOTP secret

TELEGRAM_BOT_TOKEN=...     # Optional — from @BotFather
TELEGRAM_CHAT_ID=...       # Optional — your chat ID
```

---

## Step 1: Shortlist Stocks

Run `screen_nifty50.py` to screen all Nifty 50 stocks over the last 90 days and find the best performers.

```bash
python screen_nifty50.py
```

**What it does:**

1. Fetches 90 days of 3-minute OHLCV data for all Nifty 50 stocks
2. Runs the full backtest strategy on each stock in two modes:
   - **Base** — no higher-timeframe filter
   - **HTF** — with 15-min 21-EMA trend confirmation
3. Picks the more profitable config (Base vs HTF) per stock
4. Ranks all stocks by Net P&L and outputs a shortlist

**Options:**

```bash
python screen_nifty50.py              # fetch fresh data + screen
python screen_nifty50.py --cached     # use cached CSVs (no API calls)
python screen_nifty50.py --days 180   # screen over 180 days instead of 90
```

**Output:**

- Console: ranked table with Trades, Win%, P&L, Max Drawdown, Profit Factor
- `reports/nifty50_screen.csv`: full results
- Ready-to-copy `STOCKS` and `HTF_STOCKS` lists for the live signal script

**When to run:** Weekly or bi-weekly to refresh your watchlist as market conditions change.

---

## Step 2: Generate Live Signals

Run `live_signals.py` during market hours to get real-time BUY/SHORT alerts.

```bash
python live_signals.py
```

**What it does:**

Polls Angel One every 3 minutes (at HH:MM:02) and checks all entry conditions in real-time. When a signal fires, it prints a detailed alert to the console and pushes a notification to Telegram.

**Daily lifecycle:**

| Phase | Time | Behavior |
|-------|------|----------|
| Phase 1 | 09:14 (startup) | Connect API, fetch daily candles for regime EMA, fetch 2-day 3-min warmup |
| Phase 2 | 09:15–09:59 | Silent tracking — updates indicators, no signals |
| Phase 3 | 10:00–12:00 | **Morning window — signals ON** |
| Phase 4 | 12:00–13:59 | Midday sleep — keeps indicators updated, signals blocked |
| Phase 5 | 14:00–15:00 | **Afternoon window — signals ON** |
| Phase 6 | 15:00 | Shutdown — prints day summary, exits |

**Options:**

```bash
python live_signals.py                  # default ₹50,000 capital
python live_signals.py --capital 100000 # custom capital
python live_signals.py --dry-run        # scan last cached session (no live loop)
```

**Signal output (console + Telegram):**

```
============================================================
  ▲ LONG — TATASTEEL (Metals)  [10:33:02]
============================================================
  Price  : ₹208.45
  SL     : ₹206.90  (1× ATR = ₹1.55)
  TP     : ₹210.78  (RR 1:1.5)
  Qty    : 322  (risk ₹499)
  ─────────────────────────
  VWAP   : ₹207.10
  RSI(9) : 58.3
  Vol    : 1.4× avg
  HTF EMA: ₹207.85
  Regime : ₹199.81 (Bullish)
============================================================
```

**Entry conditions (all must be true):**

| Condition | LONG | SHORT |
|-----------|------|-------|
| Regime | Close > Daily 21-EMA | Close < Daily 21-EMA |
| HTF (if enabled) | Close > 15-min 21-EMA | Close < 15-min 21-EMA |
| VWAP | Close > VWAP | Close < VWAP |
| EMA Crossover | EMA(9) crosses above EMA(21) | EMA(9) crosses below EMA(21) |
| Volume | Volume > 20-period SMA | Volume > 20-period SMA |
| RSI | 40 ≤ RSI(9) ≤ 70 | 30 ≤ RSI(9) ≤ 60 |

**Updating the watchlist:**

After running the screener (Step 1), update the `STOCKS` and `HTF_STOCKS` variables at the top of `live_signals.py` with the screener's recommended output.

---

## File Structure

```
trading/
├── live_signals.py       # Live signal generator (Step 2)
├── screen_nifty50.py     # Nifty 50 stock screener (Step 1)
├── strategy.py           # Backtrader strategy definition
├── fetch_data.py         # Angel One API client + data caching
├── batch_backtest.py     # Batch backtester for selected stocks
├── run_backtest.py       # Single-stock backtest runner
├── generate_report.py    # Report generator from backtest results
├── requirements.txt      # Python dependencies
├── .env                  # API credentials (not committed)
├── .env.example          # Credential template
├── data/                 # Cached OHLCV CSVs
└── reports/              # Backtest reports + screening results
```

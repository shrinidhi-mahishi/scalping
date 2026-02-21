# NSE Intraday Scalping Backtester

Backtest an intraday scalping strategy on 3-minute NSE candles using **backtrader**.

## Strategy Rules

| Component | Detail |
|-----------|--------|
| **Indicators** | Session VWAP, EMA(9), EMA(21), RSI(9), Volume SMA(20), ATR(14) |
| **Entry Window** | 09:15–11:30 and 13:30–15:00 (skips midday chop) |
| **Squareoff** | Force close all positions at 15:15 |
| **Long Entry** | Close > VWAP, EMA(9) crosses above EMA(21), Volume > VolSMA, RSI 40–70 |
| **Short Entry** | Close < VWAP, EMA(9) crosses below EMA(21), Volume > VolSMA, RSI 30–60 |
| **Risk:Reward** | SL = 1×ATR, Target = 1.25×ATR (bracket, no trailing) |

## Quick Start (Sample Data — No API Needed)

```bash
cd trading
pip install -r requirements.txt
python run_backtest.py --sample --days 30
```

## Angel One Setup (Live Data)

1. Register at [SmartAPI Portal](https://smartapi.angelone.in/)
2. Create an app → copy your **API Key**
3. Enable TOTP → save the **base32 secret**
4. Copy `.env.example` → `.env` and fill in:

```
ANGEL_API_KEY=your_api_key
ANGEL_CLIENT_ID=your_client_id
ANGEL_PASSWORD=your_password
ANGEL_TOTP_SECRET=your_totp_secret
```

5. Fetch and backtest:

```bash
python run_backtest.py --fetch RELIANCE --days 30
```

## CLI Options

```
--sample              Use generated sample data
--csv FILE            Load from CSV (datetime,open,high,low,close,volume)
--fetch SYMBOL        Fetch from Angel One (RELIANCE, INFY, SBIN, etc.)
--days N              Trading days of data (default: 30)
--cash AMOUNT         Starting capital (default: 100000)
--qty N               Shares per trade (default: 10)
--commission RATE     Commission rate (default: 0.0005 = 0.05%)
--plot                Show candlestick chart after backtest
--save-csv PATH       Save fetched/generated data to CSV
--quiet               Suppress trade-by-trade logs
```

## Examples

```bash
# Quick test with sample data + chart
python run_backtest.py --sample --plot

# Fetch RELIANCE data, save it, then backtest
python run_backtest.py --fetch RELIANCE --days 20 --save-csv data/reliance.csv

# Re-run from saved CSV with different capital
python run_backtest.py --csv data/reliance.csv --cash 500000 --qty 50

# Quiet mode for clean summary
python run_backtest.py --sample --days 60 --quiet
```

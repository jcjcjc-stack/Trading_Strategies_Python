# Trading Strategies Python

Python backtesting playground for comparing technical trading strategies on a sample asset series.

## Included Strategies

- Bollinger Band
- Donchian Channel Breakout
- MACD
- MACD with Bollinger confirmation
- Moving average crossover
- RSI mean reversion
- Trend RSI pullback
- Z-score mean reversion

## Project Layout

- `run_strategy_tests.py` runs all strategies and prints a comparison table.
- `connectors/` contains market data connectors.
- `strategies/` contains individual strategy backtests.
- `analytics/` contains result summary metrics.
- `data/` contains asset loading and simulation helpers.
- `risk/` contains risk-management modules.

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run against the default Monte Carlo sample:

```bash
python run_strategy_tests.py
```

Run with interactive prompts:

```bash
python run_custom_strategy_tests.py
```

The interactive runner offers presets first, then a custom mode. In custom mode, type `?` at any prompt for examples.

Run against Yahoo Finance data:

```bash
python run_strategy_tests.py --source yfinance --symbol SPY --period 1y --interval 1d
```

Yahoo Finance uses adjusted close as `price` when available, and falls back to close when adjusted close is unavailable.

Run against Binance public candle data:

```bash
python run_strategy_tests.py --source binance --symbol BTCUSDT --interval 1d --limit 1000
```

CAGR uses ACT/ACT elapsed years. Sharpe and annualized volatility infer periods/year from the data, or you can override it:

```bash
python run_strategy_tests.py --source binance --symbol BTCUSDT --periods-per-year 365
```

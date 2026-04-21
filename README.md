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
- `strategies/` contains individual strategy backtests.
- `analytics/` contains result summary metrics.
- `data/` contains asset loading and simulation helpers.
- `risk/` contains risk-management modules.

## Run

```bash
python run_strategy_tests.py
```

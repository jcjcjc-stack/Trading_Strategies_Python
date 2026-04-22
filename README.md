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

- `run_backtests.py` runs the strategy comparison with interactive presets and prompts.
- `connectors/` contains market data connectors.
- `strategies/` contains individual strategy backtests.
- `analytics/` contains result summary metrics and backtest reporting logic.
- `data/` contains asset loading and simulation helpers.
- `risk_management/` contains risk-management models.

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run with interactive prompts:

```bash
python run_backtests.py
```

The interactive runner offers presets first, then a custom mode. In custom mode, type `?` at any prompt for examples.

Use the Yahoo Finance preset or custom mode for stock and ETF data.

Yahoo Finance uses adjusted close as `price` when available, and falls back to close when adjusted close is unavailable.

Use the Binance presets or custom mode for public candle data.

CAGR uses ACT/ACT elapsed years. Sharpe and annualized volatility infer periods/year from the data, or you can override it in custom mode.

## Future Ideas

- More risk-management models.
- Strategy parameter presets.
- Charts and saved reports.
- Transaction costs, slippage, and fees.
- Tests for strategies, metrics, and connectors.
- More data sources and walk-forward testing.

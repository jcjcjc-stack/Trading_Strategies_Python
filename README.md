# Trading Strategies Python

Python playground for backtesting trading strategies and building research datasets for factor models.

## Included Strategies

- Bollinger Band
- Crossover EMA
- Crossover SMA
- Donchian Channel Breakout
- MACD
- MACD with Bollinger confirmation
- RSI mean reversion
- Trend RSI pullback
- Z-score mean reversion

## Project Layout

- `run_backtests.py` runs the strategy comparison with interactive presets and prompts.
- `connectors/` contains market data connectors.
- `strategies/` contains individual strategy backtests.
- `analytics/` contains result summary metrics and backtest reporting logic.
- `research/optimization/` contains walk-forward hyperparameter tuning and saved tuned parameter outputs.
- `research_data/` contains research/factor data helpers such as WRDS access.
- `risk_management/` contains risk-management models.

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

First, tune strategy hyperparameters:

```bash
python run_tuning.py
```

The tuning runner writes only the final selected parameters to `research/optimization/tuned_hyperparameters.txt`.
It also writes tuning notes to `research/optimization/tuned_hyperparameters_metadata.txt`, including the data source, symbol, interval, row count, date range, walk-forward sizes, and tuning objective.

Then run backtests:

```bash
python run_backtests.py
```

Both runners offer presets first, then a custom mode. In custom mode, type `?` at any prompt for examples.

Backtests always read tuned strategy parameters from `research/optimization/tuned_hyperparameters.txt`. Before the strategy table, the report checks that the tuned ticker and interval match the selected backtest data, then asks you to confirm the tune.

## Data Sources

Backtest price data:

Use the Yahoo Finance preset or custom mode for stock and ETF data.

Yahoo Finance uses adjusted close as `price` when available, and falls back to close when adjusted close is unavailable.

Use the Binance presets or custom mode for public candle data.

Research data:

WRDS is kept under `research_data/` for factor datasets, CAPM research, fundamentals, and future ML features. WRDS access requires a WRDS account and may prompt for credentials or MFA approval.

CAGR uses ACT/ACT elapsed years. Sharpe and annualized volatility infer periods/year from the data.

## Future Ideas

- More risk-management models.
- Strategy parameter presets.
- Charts and saved reports.
- Transaction costs, slippage, and fees.
- Tests for strategies, metrics, and connectors.
- WRDS factor research, CAPM models, and predictive ML models.

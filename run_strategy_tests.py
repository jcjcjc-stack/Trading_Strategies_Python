import argparse

from connectors.binance_connector import load_binance_klines
from connectors.yfinance_connector import load_yfinance_asset
from data.asset_monte_carlo import load_asset
from strategies.strategy_bollinger_band import backtest_bollinger_band
from strategies.strategy_donchian_channel_breakout import backtest_donchian_channel_breakout
from strategies.strategy_macd import backtest_macd
from strategies.strategy_macd_bollinger_confirmation import backtest_macd_bollinger_confirmation
from strategies.strategy_ma_cross import backtest_ma_cross
from strategies.strategy_rsi_mean_reversion import backtest_rsi_mean_reversion
from strategies.strategy_trend_rsi_pullback import backtest_trend_rsi_pullback
from strategies.strategy_z_score_mean_reversion import backtest_z_score_mean_reversion
from analytics.results_metrics import summarize_strategy_results


STRATEGIES = {
    "Bollinger Band": backtest_bollinger_band,
    "Donchian Channel Breakout": backtest_donchian_channel_breakout,
    "MACD": backtest_macd,
    "MACD Bollinger Confirmation": backtest_macd_bollinger_confirmation,
    "MA Cross": backtest_ma_cross,
    "RSI Mean Reversion": backtest_rsi_mean_reversion,
    "Trend RSI Pullback": backtest_trend_rsi_pullback,
    "Z Score Mean Reversion": backtest_z_score_mean_reversion,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run backtests across available strategies.")
    parser.add_argument(
        "--source",
        choices=["monte-carlo", "yfinance", "binance"],
        default="monte-carlo",
        help="Market data source to test against.",
    )
    parser.add_argument("--symbol", help="Ticker or trading pair, such as SPY or BTCUSDT.")
    parser.add_argument("--start", help="Start date, such as 2024-01-01.")
    parser.add_argument("--end", help="End date, such as 2025-01-01.")
    parser.add_argument("--period", default="1y", help="Yahoo Finance lookback period.")
    parser.add_argument("--interval", default="1d", help="Data interval, such as 1d or 1h.")
    parser.add_argument("--limit", type=int, default=1000, help="Binance candle limit.")
    parser.add_argument(
        "--periods-per-year",
        type=float,
        help="Override periods/year for Sharpe and annualized volatility.",
    )

    return parser.parse_args()


def load_test_asset(args):
    if args.source == "yfinance":
        return load_yfinance_asset(
            symbol=args.symbol or "SPY",
            start=args.start,
            end=args.end,
            period=args.period,
            interval=args.interval,
        )

    if args.source == "binance":
        return load_binance_klines(
            symbol=args.symbol or "BTCUSDT",
            interval=args.interval,
            limit=args.limit,
            start_time=args.start,
            end_time=args.end,
        )

    return load_asset()


def run_tests(args=None):
    if args is None:
        args = parse_args()

    asset = load_test_asset(args)

    print()
    print("Asset preview:")
    print(asset.head())
    print()

    print("Strategy results:")
    header = (
        f"{'Strategy':<32}"
        f"{'BuyHold':>10}"
        f"{'Strategy':>10}"
        f"{'Diff':>10}"
        f"{'Max DD':>10}"
        f"{'Long DD':>9}"
        f"{'Sharpe':>9}"
        f"{'Ann Vol':>10}"
        f"{'CAGR':>10}"
        f"{'Trades':>8}"
    )
    print(header)
    print("-" * len(header))

    strategy_results = []

    for strategy_name, backtest_function in STRATEGIES.items():
        results = backtest_function(asset)
        metrics = summarize_strategy_results(
            results,
            periods_per_year=args.periods_per_year,
        )
        strategy_results.append((strategy_name, metrics))

    strategy_results = sorted(
        strategy_results,
        key=lambda item: item[1]["strategy_return"],
        reverse=True,
    )

    for strategy_name, metrics in strategy_results:

        print(
            f"{strategy_name:<32}"
            f"{metrics['buy_hold_return']:>10.2%}"
            f"{metrics['strategy_return']:>10.2%}"
            f"{metrics['strategy_return'] - metrics['buy_hold_return']:>10.2%}"
            f"{metrics['max_drawdown']:>10.2%}"
            f"{metrics['longest_drawdown']:>9}"
            f"{metrics['sharpe_ratio']:>9.2f}"
            f"{metrics['annualized_volatility']:>10.2%}"
            f"{metrics['cagr']:>10.2%}"
            f"{metrics['number_of_trades']:>8}"
        )
    print("-" * len(header))    

    periods_per_year = strategy_results[0][1]["periods_per_year"]
    act_act_years = strategy_results[0][1]["act_act_years"]
    print(f"Sharpe/Vol periods/year: {periods_per_year:.2f}")
    print(f"CAGR ACT/ACT years: {act_act_years:.4f}")
    print()

if __name__ == "__main__":
    run_tests()

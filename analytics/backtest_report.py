import argparse
import math
from datetime import datetime

from connectors.binance import load_binance_klines
from connectors.yfinance import load_yfinance_asset
from strategies.bollinger_bands import backtest_bollinger_band
from strategies.crossover_ema import backtest_crossover_ema
from strategies.crossover_sma import backtest_crossover_sma
from strategies.donchian_breakout import backtest_donchian_channel_breakout
from strategies.macd import backtest_macd
from strategies.macd_bollinger_confirmation import backtest_macd_bollinger_confirmation
from strategies.rsi_mean_reversion import backtest_rsi_mean_reversion
from strategies.trend_rsi_pullback import backtest_trend_rsi_pullback
from strategies.zscore_mean_reversion import backtest_z_score_mean_reversion
from analytics.performance_metrics import summarize_strategy_results
from research.optimization.rolling import (
    DEFAULT_TUNED_PARAMETERS_FILE,
    load_tuned_parameters,
    load_tuning_metadata,
)


STRATEGIES = {
    "Bollinger Band": backtest_bollinger_band,
    "Crossover EMA": backtest_crossover_ema,
    "Crossover SMA": backtest_crossover_sma,
    "Donchian Channel Breakout": backtest_donchian_channel_breakout,
    "MACD": backtest_macd,
    "MACD Bollinger Confirmation": backtest_macd_bollinger_confirmation,
    "RSI Mean Reversion": backtest_rsi_mean_reversion,
    "Trend RSI Pullback": backtest_trend_rsi_pullback,
    "Z Score Mean Reversion": backtest_z_score_mean_reversion,
}

PARAMETER_GRIDS = {
    "Bollinger Band": {
        "window": [10, 20, 30],
        "num_std": [1.5, 2.0, 2.5],
    },
    "Crossover EMA": {
        "fast_window": [8, 12, 16],
        "slow_window": [21, 26, 34],
    },
    "Crossover SMA": {
        "fast_window": [10, 20, 30],
        "slow_window": [50, 100, 150],
    },
    "Donchian Channel Breakout": {
        "entry_window": [20, 40, 60],
        "exit_window": [10, 20, 30],
    },
    "MACD": {
        "fast": [8, 12],
        "slow": [21, 26, 34],
        "signal": [9],
    },
    "MACD Bollinger Confirmation": {
        "fast": [8, 12],
        "slow": [21, 26],
        "signal": [9],
        "bb_window": [20, 30],
        "num_std": [2.0],
    },
    "RSI Mean Reversion": {
        "rsi_window": [10, 14, 21],
        "oversold": [25, 30],
        "overbought": [70, 75],
        "exit_level": [50],
    },
    "Trend RSI Pullback": {
        "trend_window": [100, 150, 200],
        "rsi_window": [10, 14],
        "long_rsi_entry": [35, 40],
        "long_rsi_exit": [55, 60],
        "short_rsi_entry": [60, 65],
        "short_rsi_exit": [40, 45],
    },
    "Z Score Mean Reversion": {
        "window": [10, 20, 30],
        "entry_z": [1.5, 2.0, 2.5],
        "exit_z": [0.0, 0.5],
    },
}

PERCENT_FORMAT = "10.2%"
RATIO_FORMAT = "9.2f"
COUNT_FORMAT = "8.0f"
LONG_DD_FORMAT = "9.2f"


def format_cagr(metrics):
    if metrics.get("cagr_is_unstable"):
        return f"{'unstable':>10}"

    value = metrics.get("cagr")

    if value is None:
        return f"{'n/a':>10}"

    try:
        if not math.isfinite(value):
            return f"{'n/a':>10}"
    except TypeError:
        return f"{'n/a':>10}"

    return f"{value:{PERCENT_FORMAT}}"


def format_metadata_value(value):
    return "n/a" if value in (None, "") else value


def format_metadata_datetime(value):
    if value in (None, ""):
        return "n/a"

    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return value

    return parsed.strftime("%Y-%m-%d %H:%M UTC")


def selected_symbol(args):
    if args.source == "yfinance":
        return (args.symbol or "SPY").upper()

    if args.source == "binance":
        return (args.symbol or "BTCUSDT").upper()

    return (args.symbol or "n/a").upper()


def selected_interval(args):
    return args.interval


def format_metadata_date(value):
    if value in (None, ""):
        return "n/a"

    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return value

    return parsed.strftime("%Y-%m-%d")


def print_tune_check(metadata, args):
    data = metadata["data"]
    rolling = metadata["rolling"]
    tuned_symbol = str(format_metadata_value(data["symbol"])).upper()
    tuned_interval = format_metadata_value(data["interval"])
    backtest_symbol = selected_symbol(args)
    backtest_interval = selected_interval(args)
    is_match = tuned_symbol == backtest_symbol and tuned_interval == backtest_interval

    print("-" * 72)
    print(f"Tune check: {'OK' if is_match else 'MISMATCH'}")

    if is_match:
        print(f"Backtest matches tuned data: {backtest_symbol} {backtest_interval}")
    else:
        print(f"Tuned on : {tuned_symbol} {tuned_interval}")
        print(f"Backtest : {backtest_symbol} {backtest_interval}")

    print(
        "Tuned period: "
        f"{format_metadata_date(data['first_timestamp'])} -> "
        f"{format_metadata_date(data['last_timestamp'])}"
    )
    print(
        "Rolling validation: "
        f"train={rolling['train_size']}, "
        f"validation={rolling['validation_size']}, "
        f"test={rolling['test_size']}"
    )
    print(f"Objective   : {rolling['objective']}")
    print()

    if not is_match:
        raise SystemExit("Run python run_tuning.py for this backtest data.")


def confirm_tuned_data():
    while True:
        value = input("Use this tune? y/n [y]: ").strip().lower()
        value = value or "y"

        if value == "y":
            return

        if value == "n":
            raise SystemExit("Stopped. Run python run_tuning.py to create the tune you want.")

        print("Choose y or n.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run backtests across available strategies.")
    parser.add_argument(
        "--source",
        choices=["yfinance", "binance"],
        default="yfinance",
        help="Market data source to test against.",
    )
    parser.add_argument("--symbol", help="Ticker or trading pair, such as SPY or BTCUSDT.")
    parser.add_argument("--start", help="Start date, such as 2024-01-01.")
    parser.add_argument("--end", help="End date, such as 2025-01-01.")
    parser.add_argument("--period", default="1y", help="Yahoo Finance lookback period.")
    parser.add_argument("--interval", default="1d", help="Data interval, such as 1d or 1h.")
    parser.add_argument("--limit", type=int, help="Binance latest-candle limit.")

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

    raise ValueError(f"Unsupported data source: {args.source}")


def run_tests(args=None):
    if args is None:
        args = parse_args()

    tuned_parameters = load_tuned_parameters()

    try:
        metadata = load_tuning_metadata()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Tuning metadata is missing. Run python run_tuning.py before "
            "run_backtests.py so the backtest can confirm which data created "
            "the tuned parameters."
        ) from exc

    print_tune_check(metadata, args)
    confirm_tuned_data()

    missing_strategies = set(STRATEGIES) - set(tuned_parameters)
    if missing_strategies:
        missing = ", ".join(sorted(missing_strategies))
        raise ValueError(
            f"{DEFAULT_TUNED_PARAMETERS_FILE} is missing tuned params for: {missing}. "
            "Run python run_tuning.py before run_backtests.py."
        )

    asset = load_test_asset(args)

    print()
    print("Asset preview first rows:")
    print(asset.head().round(2))
    print()
    print("Asset preview latest rows:")
    print(asset.tail().round(2))
    print()

    print(f"Strategy results using tuned parameters for {selected_symbol(args)} ({args.source}):")
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
        results = backtest_function(asset, **tuned_parameters[strategy_name])
        metrics = summarize_strategy_results(results)
        strategy_results.append((strategy_name, metrics))

    strategy_results = sorted(
        strategy_results,
        key=lambda item: item[1]["strategy_return"],
        reverse=True,
    )

    for strategy_name, metrics in strategy_results:
        diff_return = metrics["strategy_return"] - metrics["buy_hold_return"]

        print(
            f"{strategy_name:<32}"
            f"{metrics['buy_hold_return']:{PERCENT_FORMAT}}"
            f"{metrics['strategy_return']:{PERCENT_FORMAT}}"
            f"{diff_return:{PERCENT_FORMAT}}"
            f"{metrics['max_drawdown']:{PERCENT_FORMAT}}"
            f"{metrics['longest_drawdown']:{LONG_DD_FORMAT}}"
            f"{metrics['sharpe_ratio']:{RATIO_FORMAT}}"
            f"{metrics['annualized_volatility']:{PERCENT_FORMAT}}"
            f"{format_cagr(metrics)}"
            f"{metrics['number_of_trades']:{COUNT_FORMAT}}"
        )
    print("-" * len(header))    

    periods_per_year = strategy_results[0][1]["periods_per_year"]
    act_act_years = strategy_results[0][1]["act_act_years"]
    print(f"Sharpe/Vol periods/year: {periods_per_year:.2f}")
    print(f"CAGR ACT/ACT years: {act_act_years:.2f}")
    print()

if __name__ == "__main__":
    run_tests()

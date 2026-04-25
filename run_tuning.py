import argparse
import sys
from argparse import Namespace
from datetime import UTC, datetime


DEFAULT_TUNED_PARAMETERS_FILE = "research/optimization/tuned_hyperparameters.txt"

HELP = {
    "source": "Use yfinance for stocks/ETFs, or binance for crypto pairs.",
    "symbol_yfinance": "Examples: SPY, AAPL, MSFT, QQQ.",
    "symbol_binance": "Examples: BTCUSDC, ETHUSDC, SOLUSDC.",
    "period": "Yahoo examples: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max.",
    "interval_yfinance": "Yahoo examples: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo.",
    "interval_binance": "Binance examples: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M.",
    "date": "Use yyyy-mm-dd, for example 2024-01-01. Leave blank to skip.",
    "limit": "Number of candles to request from Binance. Common range: 100 to 1000.",
}

PRESETS = {
    "1": {
        "label": "BTCUSDC daily, 1000 candles",
        "args": {
            "source": "binance",
            "symbol": "BTCUSDC",
            "interval": "1d",
            "limit": 1000,
        },
    },
    "2": {
        "label": "BTCUSDC hourly, 1000 candles",
        "args": {
            "source": "binance",
            "symbol": "BTCUSDC",
            "interval": "1h",
            "limit": 1000,
        },
    },
    "3": {
        "label": "SPY daily, 1 year",
        "args": {
            "source": "yfinance",
            "symbol": "SPY",
            "period": "1y",
            "interval": "1d",
        },
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune strategy hyperparameters and save them to a text file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        choices=["yfinance", "binance"],
        default="binance",
        help="Market data source used for tuning.",
    )
    parser.add_argument("--symbol", default="BTCUSDC", help="Ticker or trading pair, such as SPY or BTCUSDC.")
    parser.add_argument("--start", help="Start date, such as 2024-01-01.")
    parser.add_argument("--end", help="End date, such as 2025-01-01.")
    parser.add_argument("--period", default="1y", help="Yahoo Finance lookback period.")
    parser.add_argument("--interval", default="1d", help="Data interval, such as 1d or 1h.")
    parser.add_argument("--limit", type=int, default=1000, help="Binance candle limit.")
    parser.add_argument(
        "--train-size",
        type=float,
        default=0.6,
        help="Fraction of rows used for each training window.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of rows used for each test window.",
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=0.05,
        help="Fraction of rows to move forward between folds.",
    )
    parser.add_argument(
        "--objective",
        choices=["sharpe_ratio", "max_drawdown"],
        default="sharpe_ratio",
        help="Training metric used to choose hyperparameters.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_TUNED_PARAMETERS_FILE),
        help="Text file that receives tuned hyperparameters.",
    )

    return parser.parse_args()


def default_args(**overrides):
    values = {
        "source": "binance",
        "symbol": "BTCUSDC",
        "start": None,
        "end": None,
        "period": "1y",
        "interval": "1d",
        "limit": 1000,
        "train_size": 0.6,
        "test_size": 0.2,
        "step_size": 0.05,
        "objective": "sharpe_ratio",
        "output": DEFAULT_TUNED_PARAMETERS_FILE,
    }
    values.update(overrides)
    return Namespace(**values)


def ask(prompt, default=None, help_key=None):
    suffix = f" [{default}]" if default not in (None, "") else ""

    while True:
        value = input(f"{prompt}{suffix}: ").strip()

        if value == "?" and help_key:
            print(HELP[help_key])
            continue

        return value or default


def ask_choice(prompt, choices, default, help_key=None):
    choices_text = "/".join(choices)

    while True:
        value = ask(f"{prompt} ({choices_text})", default, help_key)
        value = value.lower() if isinstance(value, str) else value

        if value in choices:
            return value

        print(f"Choose one of: {choices_text}")


def ask_int(prompt, default, help_key=None):
    while True:
        value = ask(prompt, str(default), help_key)

        try:
            return int(value)
        except ValueError:
            print("Enter a whole number.")


def ask_float(prompt, default):
    while True:
        value = ask(prompt, str(default))

        try:
            return float(value)
        except ValueError:
            print("Enter a number.")


def ask_fraction(prompt, default):
    while True:
        value = ask_float(prompt, default)

        if 0 < value < 1:
            return value

        print("Enter a decimal between 0 and 1.")


def choose_preset():
    print()
    print("Choose setup")

    for key, preset in PRESETS.items():
        print(f"{key}. {preset['label']}")

    print("Type ? at any custom prompt for examples.")
    print()

    choice = input("Choose preset number, or press Enter for Custom: ").strip()

    if choice == "":
        return "custom"

    if choice in PRESETS:
        return choice

    print("Invalid choice. Using Custom.")
    return "custom"


def ask_tuning_args(args):
    args.train_size = ask_fraction("Train window fraction", 0.6)
    args.test_size = ask_fraction("Test window fraction", 0.2)
    args.step_size = ask_fraction("Step forward fraction", 0.05)
    args.objective = ask_choice(
        "Tuning objective",
        choices=["sharpe_ratio", "max_drawdown"],
        default="sharpe_ratio",
    )
    args.output = ask("Output file", DEFAULT_TUNED_PARAMETERS_FILE)
    return args


def build_custom_args():
    source = ask_choice(
        "Data source",
        choices=["yfinance", "binance"],
        default="binance",
        help_key="source",
    )

    args = default_args(source=source)

    if source == "yfinance":
        args.symbol = ask("Yahoo ticker", "SPY", "symbol_yfinance")
        args.period = ask("Yahoo period", "1y", "period")
        args.interval = ask("Yahoo interval", "1d", "interval_yfinance")
        args.start = ask("Start date optional", help_key="date")
        args.end = ask("End date optional", help_key="date")

    elif source == "binance":
        args.symbol = ask("Binance pair", "BTCUSDC", "symbol_binance")
        args.interval = ask("Binance interval", "1d", "interval_binance")
        args.limit = ask_int("Candle limit", 1000, "limit")
        args.start = ask("Start date optional", help_key="date")
        args.end = ask("End date optional", help_key="date")

    return ask_tuning_args(args)


def build_args():
    choice = choose_preset()

    if choice == "custom":
        return build_custom_args()

    args = default_args(**PRESETS[choice]["args"])
    return ask_tuning_args(args)


def tune_strategies(args):
    from analytics.backtest_report import PARAMETER_GRIDS, STRATEGIES, load_test_asset
    from research.optimization.walk_forward import (
        save_tuned_parameters,
        save_tuning_metadata,
        select_final_parameters,
        walk_forward_validate,
    )

    asset = load_test_asset(args)
    tuned_parameters = {}

    print()
    print(f"Loaded {len(asset)} rows for tuning.")
    print(f"Saving tuned parameters to: {args.output}")
    print()

    for strategy_name, backtest_function in STRATEGIES.items():
        print(f"Tuning {strategy_name}...")
        validation = walk_forward_validate(
            asset,
            backtest_function,
            PARAMETER_GRIDS[strategy_name],
            train_size=args.train_size,
            test_size=args.test_size,
            step_size=args.step_size,
            objective=args.objective,
        )
        tuned_parameters[strategy_name] = select_final_parameters(validation["folds"])

    output_path = save_tuned_parameters(tuned_parameters, args.output)
    metadata = build_tuning_metadata(args, asset, output_path)
    metadata_path = save_tuning_metadata(metadata, output_path=output_path)
    return tuned_parameters, output_path, metadata_path


def build_tuning_metadata(args, asset, output_path):
    index = asset.index
    data_start = index[0] if len(index) else None
    data_end = index[-1] if len(index) else None

    return {
        "created_at_utc": datetime.now(UTC).replace(microsecond=0),
        "tuned_parameters_file": output_path,
        "data": {
            "source": args.source,
            "symbol": args.symbol,
            "start_arg": args.start,
            "end_arg": args.end,
            "period": args.period,
            "interval": args.interval,
            "limit": args.limit,
            "rows": len(asset),
            "first_timestamp": data_start,
            "last_timestamp": data_end,
        },
        "walk_forward": {
            "train_size": args.train_size,
            "test_size": args.test_size,
            "step_size": args.step_size,
            "objective": args.objective,
        },
        "run_backtests_note": "run_backtests.py always reads this tuned parameter file.",
    }


def print_summary(tuned_parameters, output_path, metadata_path):
    print()
    print("Tuned hyperparameters saved.")
    print(f"Output file: {output_path}")
    print(f"Tuning metadata: {metadata_path}")
    print("run_backtests.py will use this tune file.")
    print()

    for strategy_name, params in tuned_parameters.items():
        params_text = ", ".join(f"{key}={value}" for key, value in params.items())
        print(f"{strategy_name}: {params_text}")

    print()


if __name__ == "__main__":
    args = parse_args() if len(sys.argv) > 1 else build_args()
    tuned_parameters, output_path, metadata_path = tune_strategies(args)
    print_summary(tuned_parameters, output_path, metadata_path)

from argparse import Namespace

from analytics.backtest_report import run_tests


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


def default_args(**overrides):
    values = {
        "source": "yfinance",
        "symbol": "SPY",
        "start": None,
        "end": None,
        "period": "1y",
        "interval": "1d",
        "limit": 1000,
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


def build_custom_args():
    source = ask_choice(
        "Data source",
        choices=["yfinance", "binance"],
        default="yfinance",
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

    return args


def build_args():
    choice = choose_preset()

    if choice == "custom":
        return build_custom_args()

    args = default_args(**PRESETS[choice]["args"])
    return args


if __name__ == "__main__":
    run_tests(build_args())

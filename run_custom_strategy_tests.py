from argparse import Namespace

from run_strategy_tests import run_tests


HELP = {
    "source": "Use yfinance for stocks/ETFs, binance for crypto pairs, or monte-carlo for sample data.",
    "symbol_yfinance": "Examples: SPY, AAPL, MSFT, QQQ.",
    "symbol_binance": "Examples: BTCUSDT, ETHUSDT, SOLUSDT.",
    "period": "Yahoo examples: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max.",
    "interval_yfinance": "Yahoo examples: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo.",
    "interval_binance": "Binance examples: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M.",
    "date": "Use yyyy-mm-dd, for example 2024-01-01. Leave blank to skip.",
    "limit": "Number of candles to request from Binance. Common range: 100 to 1000.",
    "periods_per_year": "Optional. Use 252 for daily stocks, 365 for daily crypto, 8760 for hourly crypto.",
}

PRESETS = {
    "1": {
        "label": "SPY daily, 1 year",
        "args": {
            "source": "yfinance",
            "symbol": "SPY",
            "period": "1y",
            "interval": "1d",
        },
    },
    "2": {
        "label": "BTCUSDT daily, 1000 candles",
        "args": {
            "source": "binance",
            "symbol": "BTCUSDT",
            "interval": "1d",
            "limit": 1000,
        },
    },
    "3": {
        "label": "BTCUSDT hourly, 1000 candles",
        "args": {
            "source": "binance",
            "symbol": "BTCUSDT",
            "interval": "1h",
            "limit": 1000,
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
        "periods_per_year": None,
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


def ask_float_or_none(prompt, help_key=None):
    value = ask(prompt, help_key=help_key)

    if value in (None, ""):
        return None

    try:
        return float(value)
    except ValueError:
        print("Invalid number. Leaving this blank.")
        return None


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
        choices=["yfinance", "binance", "monte-carlo"],
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
        args.symbol = ask("Binance pair", "BTCUSDT", "symbol_binance")
        args.interval = ask("Binance interval", "1d", "interval_binance")
        args.limit = ask_int("Candle limit", 1000, "limit")
        args.start = ask("Start date optional", help_key="date")
        args.end = ask("End date optional", help_key="date")

    args.periods_per_year = ask_float_or_none(
        "Sharpe/Vol periods per year override optional",
        "periods_per_year",
    )

    return args


def build_args():
    choice = choose_preset()

    if choice == "custom":
        return build_custom_args()

    return default_args(**PRESETS[choice]["args"])


if __name__ == "__main__":
    run_tests(build_args())

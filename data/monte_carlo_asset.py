import numpy as np
import pandas as pd


def generate_monte_carlo_asset(
    start_price=100.0,
    periods=252 * 3,
    mean_return=0.0002,
    volatility=0.01,
    seed=42,
    start_date="2025-01-01",
):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start_date, periods=periods)

    daily_returns = rng.normal(loc=mean_return, scale=volatility, size=periods)
    price = start_price * np.exp(np.cumsum(daily_returns))

    open_price = np.empty(periods)
    open_price[0] = start_price
    open_price[1:] = price[:-1]

    intraday_range = np.abs(rng.normal(loc=0.005, scale=0.003, size=periods))
    high = np.maximum(open_price, price) * (1 + intraday_range)
    low = np.minimum(open_price, price) * (1 - intraday_range)
    volume = rng.integers(low=100_000, high=1_000_000, size=periods)

    asset = pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "price": price,
            "volume": volume,
        },
        index=dates,
    )
    asset.index.name = "date"

    return asset


def load_asset():
    return generate_monte_carlo_asset()


if __name__ == "__main__":
    asset = load_asset()
    print(asset.head())
    print(asset.tail())

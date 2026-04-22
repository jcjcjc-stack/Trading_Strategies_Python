import json
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd


BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


def to_milliseconds(value):
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return int(value)

    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return int(parsed.timestamp() * 1000)


def load_binance_klines(
    symbol="BTCUSDT",
    interval="1d",
    limit=1000,
    start_time=None,
    end_time=None,
):
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": int(limit),
    }

    if start_time is not None:
        params["startTime"] = to_milliseconds(start_time)

    if end_time is not None:
        params["endTime"] = to_milliseconds(end_time)

    url = f"{BINANCE_KLINES_URL}?{urlencode(params)}"

    with urlopen(url, timeout=30) as response:
        rows = json.loads(response.read().decode("utf-8"))

    if not rows:
        raise ValueError(f"No Binance data returned for symbol: {symbol}")

    numeric_columns = ["open", "high", "low", "price", "volume"]
    asset = pd.DataFrame(
        [row[:6] for row in rows],
        columns=["date", *numeric_columns],
    )
    asset["date"] = pd.to_datetime(asset["date"], unit="ms", utc=True)
    asset = asset.set_index("date")
    asset[numeric_columns] = asset[numeric_columns].astype(float)

    return asset

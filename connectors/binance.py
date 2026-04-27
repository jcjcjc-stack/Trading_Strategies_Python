import json
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd


BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
BINANCE_MAX_KLINES_LIMIT = 1000


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
    symbol="BTCUSDC",
    interval="1d",
    limit=1000,
    start_time=None,
    end_time=None,
):
    symbol = symbol.upper()
    start_ms = to_milliseconds(start_time)
    end_ms = to_milliseconds(end_time)

    base_params = {
        "symbol": symbol.upper(),
        "interval": interval,
    }

    if start_ms is None:
        total_limit = _validate_limit(limit)
        rows = _load_latest_klines(base_params, total_limit, end_ms)
    else:
        rows = _load_forward_klines(base_params, start_ms, end_ms)

    if not rows:
        raise ValueError(f"No Binance data returned for symbol: {symbol}")

    rows.sort(key=lambda row: row[0])

    return _rows_to_asset(rows)


def _validate_limit(limit):
    total_limit = 1000 if limit is None else int(limit)
    if total_limit < 1:
        raise ValueError("Binance candle limit must be at least 1.")

    return total_limit


def _fetch_klines(params):
    url = f"{BINANCE_KLINES_URL}?{urlencode(params)}"

    with urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _load_forward_klines(base_params, start_ms, end_ms):
    rows = []
    next_start = start_ms

    while True:
        request_limit = BINANCE_MAX_KLINES_LIMIT
        params = {
            **base_params,
            "limit": request_limit,
            "startTime": next_start,
        }

        if end_ms is not None:
            params["endTime"] = end_ms

        batch = _fetch_klines(params)
        if not batch:
            break

        rows.extend(batch)

        last_open_time = batch[-1][0]
        next_start = last_open_time + 1

        if len(batch) < request_limit:
            break

    return rows


def _load_latest_klines(base_params, total_limit, end_ms):
    rows = []
    next_end = end_ms

    while len(rows) < total_limit:
        request_limit = min(BINANCE_MAX_KLINES_LIMIT, total_limit - len(rows))
        params = {
            **base_params,
            "limit": request_limit,
        }

        if next_end is not None:
            params["endTime"] = next_end

        batch = _fetch_klines(params)
        if not batch:
            break

        rows = batch + rows
        first_open_time = batch[0][0]
        next_end = first_open_time - 1

        if len(batch) < request_limit:
            break

    return rows


def _rows_to_asset(rows):
    numeric_columns = ["open", "high", "low", "price", "volume"]
    asset = pd.DataFrame(
        [row[:6] for row in rows],
        columns=["date", *numeric_columns],
    )
    asset["date"] = pd.to_datetime(asset["date"], unit="ms", utc=True)
    asset = asset.set_index("date")
    asset[numeric_columns] = asset[numeric_columns].astype(float)

    return asset

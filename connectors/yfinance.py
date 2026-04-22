def load_yfinance_asset(
    symbol="SPY",
    start=None,
    end=None,
    period="3y",
    interval="1d",
    auto_adjust=False,
    prefer_adjusted=True,
):
    try:
        import yfinance as yf
    except ImportError as error:
        raise ImportError(
            "Install yfinance before using the Yahoo Finance connector: "
            "pip install yfinance"
        ) from error

    params = {
        "start": start,
        "end": end,
    } if start or end else {
        "period": period,
    }

    data = yf.download(
        symbol,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
        **params,
    )

    if data.empty:
        raise ValueError(f"No Yahoo Finance data returned for symbol: {symbol}")

    if getattr(data.columns, "nlevels", 1) > 1:
        data.columns = data.columns.get_level_values(0)
    data.columns.name = None

    data = data.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "price",
            "Adj Close": "adjusted_price",
            "Volume": "volume",
        }
    )

    columns = ["open", "high", "low", "price", "adjusted_price", "volume"]
    asset = data[[column for column in columns if column in data]].copy()

    if "price" not in asset and "adjusted_price" in asset:
        asset["price"] = asset["adjusted_price"]

    if prefer_adjusted and "adjusted_price" in asset:
        asset["price"] = asset["adjusted_price"]

    asset = asset[["open", "high", "low", "price", "volume"]]
    asset.columns.name = None
    asset.index.name = "date"

    return asset.dropna()

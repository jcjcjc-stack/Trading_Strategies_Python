import numpy as np
import pandas as pd

# assumes dataframe = asset
# requires column: 'price'
# datetime index already set

def backtest_trend_rsi_pullback(asset, price_col='price',
                                trend_window=200, rsi_window=14,
                                long_rsi_entry=40, long_rsi_exit=60,
                                short_rsi_entry=60, short_rsi_exit=40):

    df = asset.copy()

    # ---------- Indicators ----------
    df['trend_ma'] = df[price_col].rolling(window=trend_window).mean()
    df['price_change'] = df[price_col].diff()
    df['gain'] = df['price_change'].clip(lower=0)
    df['loss'] = -df['price_change'].clip(upper=0)
    df['avg_gain'] = df['gain'].ewm(alpha=1 / rsi_window, adjust=False).mean()
    df['avg_loss'] = df['loss'].ewm(alpha=1 / rsi_window, adjust=False).mean()
    df['rs'] = df['avg_gain'] / df['avg_loss'].replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + df['rs']))
    df['rsi'] = df['rsi'].fillna(50)

    df['uptrend'] = df[price_col] > df['trend_ma']
    df['downtrend'] = df[price_col] < df['trend_ma']

    # ---------- Trading Signal ----------
    # Pullback logic:
    # long in uptrend when RSI recovers above long_rsi_entry
    # short in downtrend when RSI falls below short_rsi_entry
    prev_rsi = df['rsi'].shift(1)
    prev_price = df[price_col].shift(1)
    prev_trend_ma = df['trend_ma'].shift(1)

    df['long_entry'] = df['uptrend'] & (df['rsi'] > long_rsi_entry) & (prev_rsi <= long_rsi_entry)
    df['long_exit'] = (
        ((df['rsi'] > long_rsi_exit) & (prev_rsi <= long_rsi_exit)) |
        ((df[price_col] < df['trend_ma']) & (prev_price >= prev_trend_ma))
    )

    df['short_entry'] = df['downtrend'] & (df['rsi'] < short_rsi_entry) & (prev_rsi >= short_rsi_entry)
    df['short_exit'] = (
        ((df['rsi'] < short_rsi_exit) & (prev_rsi >= short_rsi_exit)) |
        ((df[price_col] > df['trend_ma']) & (prev_price <= prev_trend_ma))
    )

    # ---------- Position ----------
    df['signal_raw'] = 0
    position = 0

    for i in range(len(df)):
        if position == 1 and df['long_exit'].iloc[i]:
            position = 0
        elif position == -1 and df['short_exit'].iloc[i]:
            position = 0

        if position == 0:
            if df['long_entry'].iloc[i]:
                position = 1
            elif df['short_entry'].iloc[i]:
                position = -1

        df.iloc[i, df.columns.get_loc('signal_raw')] = position

    # For backtesting, shift positions to avoid lookahead bias.
    df['position'] = df['signal_raw'].shift(1).fillna(0).astype(int)

    # ---------- Returns ----------
    df['return'] = df[price_col].pct_change().fillna(0)
    df['strategy_return'] = df['position'] * df['return']

    # ---------- Equity Curve ----------
    df['cum_asset'] = (1 + df['return']).cumprod()
    df['cum_strategy'] = (1 + df['strategy_return']).cumprod()

    return df


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from data.asset_monte_carlo import load_asset

    asset = load_asset()
    results = backtest_trend_rsi_pullback(asset)

    print("Buy & Hold:", results['cum_asset'].iloc[-1])
    print("Trend RSI Pullback Strategy:", results['cum_strategy'].iloc[-1])
    print(results.tail())

import numpy as np
import pandas as pd

# assumes dataframe = asset
# requires column: 'price'
# datetime index already set

def backtest_bollinger_band(asset, price_col='price',
                            window=20, num_std=2):

    df = asset.copy()

    # ---------- Indicators ----------
    df['middle_band'] = df[price_col].rolling(window=window).mean()
    df['std'] = df[price_col].rolling(window=window).std()
    df['upper_band'] = df['middle_band'] + (num_std * df['std'])
    df['lower_band'] = df['middle_band'] - (num_std * df['std'])

    # ---------- Trading Signal ----------
    # Mean reversion logic:
    # long entry when price crosses below lower band
    # long exit when price crosses back above middle band
    # short entry when price crosses above upper band
    # short exit when price crosses back below middle band
    prev_price = df[price_col].shift(1)
    prev_lower = df['lower_band'].shift(1)
    prev_upper = df['upper_band'].shift(1)
    prev_middle = df['middle_band'].shift(1)

    df['long_entry'] = (df[price_col] < df['lower_band']) & (prev_price >= prev_lower)
    df['long_exit'] = (df[price_col] > df['middle_band']) & (prev_price <= prev_middle)
    df['short_entry'] = (df[price_col] > df['upper_band']) & (prev_price <= prev_upper)
    df['short_exit'] = (df[price_col] < df['middle_band']) & (prev_price >= prev_middle)

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
    from data.monte_carlo_asset import load_asset

    asset = load_asset()
    results = backtest_bollinger_band(asset)

    print("Buy & Hold:", results['cum_asset'].iloc[-1])
    print("Bollinger Band Strategy:", results['cum_strategy'].iloc[-1])
    print(results.tail())

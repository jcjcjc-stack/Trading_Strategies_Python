import numpy as np
import pandas as pd

# assumes dataframe = asset
# requires column: 'price'
# datetime index already set

def backtest_ma_cross(asset, price_col='price',
                      fast_window=20, slow_window=50):

    df = asset.copy()

    # ---------- Indicators ----------
    df['ma_fast'] = df[price_col].rolling(window=fast_window).mean()
    df['ma_slow'] = df[price_col].rolling(window=slow_window).mean()

    # ---------- Trading Signal ----------
    prev_fast = df['ma_fast'].shift(1)
    prev_slow = df['ma_slow'].shift(1)

    df['long_entry'] = (df['ma_fast'] > df['ma_slow']) & (prev_fast <= prev_slow)
    df['long_exit'] = (df['ma_fast'] < df['ma_slow']) & (prev_fast >= prev_slow)
    df['short_entry'] = df['long_exit']
    df['short_exit'] = df['long_entry']

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
    results = backtest_ma_cross(asset)

    print("Buy & Hold:", results['cum_asset'].iloc[-1])
    print("MA Cross Strategy:", results['cum_strategy'].iloc[-1])
    print(results.tail())

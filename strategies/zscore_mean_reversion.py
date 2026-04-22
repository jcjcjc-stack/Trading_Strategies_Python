import numpy as np
import pandas as pd

# assumes dataframe = asset
# requires column: 'price'
# datetime index already set

def backtest_z_score_mean_reversion(asset, price_col='price',
                                    window=20, entry_z=2.0,
                                    exit_z=0.0):

    df = asset.copy()

    # ---------- Indicators ----------
    df['rolling_mean'] = df[price_col].rolling(window=window).mean()
    df['rolling_std'] = df[price_col].rolling(window=window).std()
    df['z_score'] = (df[price_col] - df['rolling_mean']) / df['rolling_std']

    # ---------- Trading Signal ----------
    # Mean reversion logic:
    # long when z-score crosses below -entry_z, exit near exit_z
    # short when z-score crosses above entry_z, exit near exit_z
    prev_z = df['z_score'].shift(1)

    df['long_entry'] = (df['z_score'] < -entry_z) & (prev_z >= -entry_z)
    df['long_exit'] = (df['z_score'] > exit_z) & (prev_z <= exit_z)
    df['short_entry'] = (df['z_score'] > entry_z) & (prev_z <= entry_z)
    df['short_exit'] = (df['z_score'] < -exit_z) & (prev_z >= -exit_z)

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
    results = backtest_z_score_mean_reversion(asset)

    print("Buy & Hold:", results['cum_asset'].iloc[-1])
    print("Z Score Mean Reversion Strategy:", results['cum_strategy'].iloc[-1])
    print(results.tail())

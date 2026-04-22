import numpy as np
import pandas as pd

# assumes dataframe = asset
# requires column: 'price'
# datetime index already set

def backtest_donchian_channel_breakout(asset, price_col='price',
                                       entry_window=20, exit_window=10):

    df = asset.copy()

    # ---------- Indicators ----------
    df['donchian_high'] = df[price_col].rolling(window=entry_window).max().shift(1)
    df['donchian_low'] = df[price_col].rolling(window=entry_window).min().shift(1)
    df['exit_high'] = df[price_col].rolling(window=exit_window).max().shift(1)
    df['exit_low'] = df[price_col].rolling(window=exit_window).min().shift(1)

    # ---------- Trading Signal ----------
    # Breakout logic:
    # long when price breaks above prior Donchian high
    # short when price breaks below prior Donchian low
    df['long_entry'] = df[price_col] > df['donchian_high']
    df['long_exit'] = df[price_col] < df['exit_low']
    df['short_entry'] = df[price_col] < df['donchian_low']
    df['short_exit'] = df[price_col] > df['exit_high']

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
    results = backtest_donchian_channel_breakout(asset)

    print("Buy & Hold:", results['cum_asset'].iloc[-1])
    print("Donchian Channel Breakout Strategy:", results['cum_strategy'].iloc[-1])
    print(results.tail())

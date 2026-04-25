import pandas as pd
import numpy as np

# assumes dataframe = asset
# requires column: 'price'
# datetime index already set

def backtest_macd(asset, price_col='price',
                  fast=12, slow=26, signal=9):

    df = asset.copy()

    # ---------- Indicators ----------
    df['ema_fast'] = df[price_col].ewm(span=fast, adjust=False).mean()
    df['ema_slow'] = df[price_col].ewm(span=slow, adjust=False).mean()

    df['macd'] = df['ema_fast'] - df['ema_slow']
    df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
    df['histogram'] = df['macd'] - df['macd_signal']

    # ---------- Trading Signal ----------
    # Entry/exit events based on MACD crossing the signal line.
    prev_macd = df['macd'].shift(1)
    prev_signal = df['macd_signal'].shift(1)
 
    df['long_entry'] = (df['macd'] > df['macd_signal']) & (prev_macd <= prev_signal)
    df['long_exit'] = (df['macd'] < df['macd_signal']) & (prev_macd >= prev_signal)
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

import numpy as np
import pandas as pd

# assumes dataframe = asset
# requires columns: 'high', 'low', 'price', and a signal column such as 'signal_raw'
# datetime index already set

def apply_atr_stop_loss_take_profit(asset, price_col='price',
                                    high_col='high', low_col='low',
                                    signal_col='signal_raw',
                                    atr_window=14, stop_atr=2.0,
                                    take_profit_atr=3.0):

    df = asset.copy()

    # ---------- Indicators ----------
    prev_close = df[price_col].shift(1)
    df['true_range'] = np.maximum(
        df[high_col] - df[low_col],
        np.maximum(
            (df[high_col] - prev_close).abs(),
            (df[low_col] - prev_close).abs()
        )
    )
    df['atr'] = df['true_range'].ewm(alpha=1 / atr_window, adjust=False).mean()

    # ---------- Stop / Take Profit Levels ----------
    df['entry_price'] = np.nan
    df['stop_loss'] = np.nan
    df['take_profit'] = np.nan
    df['stop_loss_hit'] = False
    df['take_profit_hit'] = False
    df['signal_with_risk'] = 0

    position = 0
    entry_price = np.nan
    stop_loss = np.nan
    take_profit = np.nan

    for i in range(len(df)):
        raw_signal = df[signal_col].iloc[i]
        price = df[price_col].iloc[i]
        high = df[high_col].iloc[i]
        low = df[low_col].iloc[i]
        atr = df['atr'].iloc[i]

        if position == 0:
            if raw_signal == 1 and not np.isnan(atr):
                position = 1
                entry_price = price
                stop_loss = entry_price - (stop_atr * atr)
                take_profit = entry_price + (take_profit_atr * atr)
            elif raw_signal == -1 and not np.isnan(atr):
                position = -1
                entry_price = price
                stop_loss = entry_price + (stop_atr * atr)
                take_profit = entry_price - (take_profit_atr * atr)

        elif position == 1:
            if low <= stop_loss:
                df.iloc[i, df.columns.get_loc('stop_loss_hit')] = True
                position = 0
            elif high >= take_profit:
                df.iloc[i, df.columns.get_loc('take_profit_hit')] = True
                position = 0
            elif raw_signal <= 0:
                position = 0

        elif position == -1:
            if high >= stop_loss:
                df.iloc[i, df.columns.get_loc('stop_loss_hit')] = True
                position = 0
            elif low <= take_profit:
                df.iloc[i, df.columns.get_loc('take_profit_hit')] = True
                position = 0
            elif raw_signal >= 0:
                position = 0

        df.iloc[i, df.columns.get_loc('entry_price')] = entry_price
        df.iloc[i, df.columns.get_loc('stop_loss')] = stop_loss
        df.iloc[i, df.columns.get_loc('take_profit')] = take_profit
        df.iloc[i, df.columns.get_loc('signal_with_risk')] = position

        if position == 0:
            entry_price = np.nan
            stop_loss = np.nan
            take_profit = np.nan

    # For backtesting, shift positions to avoid lookahead bias.
    df['position'] = df['signal_with_risk'].shift(1).fillna(0).astype(int)

    # ---------- Returns ----------
    df['return'] = df[price_col].pct_change().fillna(0)
    df['strategy_return'] = df['position'] * df['return']

    # ---------- Equity Curve ----------
    df['cum_asset'] = (1 + df['return']).cumprod()
    df['cum_strategy'] = (1 + df['strategy_return']).cumprod()

    return df

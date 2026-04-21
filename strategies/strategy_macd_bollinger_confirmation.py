import numpy as np
import pandas as pd

# assumes dataframe = asset
# requires column: 'price'
# datetime index already set

def backtest_macd_bollinger_confirmation(asset, price_col='price',
                                         fast=12, slow=26, signal=9,
                                         bb_window=20, num_std=2):

    df = asset.copy()

    # ---------- Indicators ----------
    df['ema_fast'] = df[price_col].ewm(span=fast, adjust=False).mean()
    df['ema_slow'] = df[price_col].ewm(span=slow, adjust=False).mean()
    df['macd'] = df['ema_fast'] - df['ema_slow']
    df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
    df['histogram'] = df['macd'] - df['macd_signal']

    df['middle_band'] = df[price_col].rolling(window=bb_window).mean()
    df['std'] = df[price_col].rolling(window=bb_window).std()
    df['upper_band'] = df['middle_band'] + (num_std * df['std'])
    df['lower_band'] = df['middle_band'] - (num_std * df['std'])

    # ---------- Trading Signal ----------
    # Confirmation logic:
    # long when price crosses above middle band and MACD confirms bullish momentum
    # short when price crosses below middle band and MACD confirms bearish momentum
    prev_price = df[price_col].shift(1)
    prev_middle = df['middle_band'].shift(1)

    bullish_macd = df['macd'] > df['macd_signal']
    bearish_macd = df['macd'] < df['macd_signal']

    df['long_entry'] = (df[price_col] > df['middle_band']) & (prev_price <= prev_middle) & bullish_macd
    df['long_exit'] = ((df[price_col] < df['middle_band']) & (prev_price >= prev_middle)) | bearish_macd
    df['short_entry'] = (df[price_col] < df['middle_band']) & (prev_price >= prev_middle) & bearish_macd
    df['short_exit'] = ((df[price_col] > df['middle_band']) & (prev_price <= prev_middle)) | bullish_macd

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
    results = backtest_macd_bollinger_confirmation(asset)

    print("Buy & Hold:", results['cum_asset'].iloc[-1])
    print("MACD Bollinger Confirmation Strategy:", results['cum_strategy'].iloc[-1])
    print(results.tail())

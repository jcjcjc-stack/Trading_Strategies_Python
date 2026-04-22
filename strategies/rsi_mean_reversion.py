import numpy as np
import pandas as pd

# assumes dataframe = asset
# requires column: 'price'
# datetime index already set

def backtest_rsi_mean_reversion(asset, price_col='price',
                                rsi_window=14, oversold=30,
                                overbought=70, exit_level=50):

    df = asset.copy()

    # ---------- Indicators ----------
    df['price_change'] = df[price_col].diff()
    df['gain'] = df['price_change'].clip(lower=0)
    df['loss'] = -df['price_change'].clip(upper=0)
    df['avg_gain'] = df['gain'].ewm(span=rsi_window, adjust=False).mean()
    df['avg_loss'] = df['loss'].ewm(span=rsi_window, adjust=False).mean()
    df['rs'] = df['avg_gain'] / df['avg_loss'].replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + df['rs']))
    df['rsi'] = df['rsi'].fillna(50)

    # ---------- Trading Signal ----------
    # Mean reversion logic:
    # long when RSI crosses below oversold, exit when RSI crosses above exit level
    # short when RSI crosses above overbought, exit when RSI crosses below exit level
    prev_rsi = df['rsi'].shift(1)

    df['long_entry'] = (df['rsi'] < oversold) & (prev_rsi >= oversold)
    df['long_exit'] = (df['rsi'] > exit_level) & (prev_rsi <= exit_level)
    df['short_entry'] = (df['rsi'] > overbought) & (prev_rsi <= overbought)
    df['short_exit'] = (df['rsi'] < exit_level) & (prev_rsi >= exit_level)

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
    results = backtest_rsi_mean_reversion(asset)

    print("Buy & Hold:", results['cum_asset'].iloc[-1])
    print("RSI Mean Reversion Strategy:", results['cum_strategy'].iloc[-1])
    print(results.tail())

from data.asset_monte_carlo import load_asset
from strategies.strategy_bollinger_band import backtest_bollinger_band
from strategies.strategy_donchian_channel_breakout import backtest_donchian_channel_breakout
from strategies.strategy_macd import backtest_macd
from strategies.strategy_macd_bollinger_confirmation import backtest_macd_bollinger_confirmation
from strategies.strategy_ma_cross import backtest_ma_cross
from strategies.strategy_rsi_mean_reversion import backtest_rsi_mean_reversion
from strategies.strategy_trend_rsi_pullback import backtest_trend_rsi_pullback
from strategies.strategy_z_score_mean_reversion import backtest_z_score_mean_reversion
from analytics.results_metrics import summarize_strategy_results


STRATEGIES = {
    "Bollinger Band": backtest_bollinger_band,
    "Donchian Channel Breakout": backtest_donchian_channel_breakout,
    "MACD": backtest_macd,
    "MACD Bollinger Confirmation": backtest_macd_bollinger_confirmation,
    "MA Cross": backtest_ma_cross,
    "RSI Mean Reversion": backtest_rsi_mean_reversion,
    "Trend RSI Pullback": backtest_trend_rsi_pullback,
    "Z Score Mean Reversion": backtest_z_score_mean_reversion,
}


def run_tests():
    asset = load_asset()

    print("Asset preview:")
    print(asset.head())
    print()

    print("Strategy results:")
    header = (
        f"{'Strategy':<32}"
        f"{'BuyHold':>10}"
        f"{'Strategy':>10}"
        f"{'Diff':>10}"
        f"{'Max DD':>10}"
        f"{'Long DD':>9}"
        f"{'Sharpe':>9}"
        f"{'Ann Vol':>10}"
        f"{'CAGR':>10}"
        f"{'Trades':>8}"
    )
    print(header)
    print("-" * len(header))

    strategy_results = []

    for strategy_name, backtest_function in STRATEGIES.items():
        results = backtest_function(asset)
        metrics = summarize_strategy_results(results)
        strategy_results.append((strategy_name, metrics))

    strategy_results = sorted(
        strategy_results,
        key=lambda item: item[1]["strategy_return"],
        reverse=True,
    )

    for strategy_name, metrics in strategy_results:

        print(
            f"{strategy_name:<32}"
            f"{metrics['buy_hold_return']:>10.2%}"
            f"{metrics['strategy_return']:>10.2%}"
            f"{metrics['strategy_return'] - metrics['buy_hold_return']:>10.2%}"
            f"{metrics['max_drawdown']:>10.2%}"
            f"{metrics['longest_drawdown']:>9}"
            f"{metrics['sharpe_ratio']:>9.2f}"
            f"{metrics['annualized_volatility']:>10.2%}"
            f"{metrics['cagr']:>10.2%}"
            f"{metrics['number_of_trades']:>8}"
        )
    print("-" * len(header))    


if __name__ == "__main__":
    run_tests()

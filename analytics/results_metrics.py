import numpy as np


TRADING_DAYS_PER_YEAR = 252


def calculate_max_drawdown(equity_curve):
    running_peak = equity_curve.cummax()
    drawdown = (equity_curve / running_peak) - 1
    return drawdown.min()


def calculate_longest_drawdown(equity_curve):
    running_peak = equity_curve.cummax()
    in_drawdown = equity_curve < running_peak

    longest = 0
    current = 0

    for is_in_drawdown in in_drawdown:
        if is_in_drawdown:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return longest


def calculate_sharpe_ratio(returns, periods_per_year=TRADING_DAYS_PER_YEAR):
    mean_return = returns.mean()
    std_return = returns.std()

    if std_return == 0 or np.isnan(std_return):
        return np.nan

    return (mean_return / std_return) * np.sqrt(periods_per_year)


def calculate_annualized_volatility(returns, periods_per_year=TRADING_DAYS_PER_YEAR):
    return returns.std() * np.sqrt(periods_per_year)


def calculate_cagr(equity_curve, periods_per_year=TRADING_DAYS_PER_YEAR):
    if len(equity_curve) == 0:
        return np.nan

    final_value = equity_curve.iloc[-1]
    years = len(equity_curve) / periods_per_year

    if final_value <= 0 or years <= 0:
        return np.nan

    return (final_value ** (1 / years)) - 1


def count_trades(signal):
    signal_changes = signal.diff().fillna(0) != 0
    entries = signal_changes & (signal != 0)
    return entries.sum()


def summarize_strategy_results(results):
    return {
        "buy_hold_return": results["cum_asset"].iloc[-1] - 1,
        "strategy_return": results["cum_strategy"].iloc[-1] - 1,
        "max_drawdown": calculate_max_drawdown(results["cum_strategy"]),
        "longest_drawdown": calculate_longest_drawdown(results["cum_strategy"]),
        "sharpe_ratio": calculate_sharpe_ratio(results["strategy_return"]),
        "annualized_volatility": calculate_annualized_volatility(results["strategy_return"]),
        "mean": results["strategy_return"].mean(),
        "std": results["strategy_return"].std(),
        "cagr": calculate_cagr(results["cum_strategy"]),
        "number_of_trades": count_trades(results["signal_raw"]),
    }

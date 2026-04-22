import numpy as np
import pandas as pd


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


def calculate_act_act_years(index):
    if len(index) < 2:
        return np.nan

    start = pd.Timestamp(index[0])
    end = pd.Timestamp(index[-1])

    if end <= start:
        return np.nan

    years = 0.0
    timezone = start.tz

    for year in range(start.year, end.year + 1):
        year_start = pd.Timestamp(year=year, month=1, day=1, tz=timezone)
        year_end = pd.Timestamp(year=year + 1, month=1, day=1, tz=timezone)
        segment_start = max(start, year_start)
        segment_end = min(end, year_end)

        if segment_end > segment_start:
            years += (
                (segment_end - segment_start).total_seconds()
                / (year_end - year_start).total_seconds()
            )

    return years


def calculate_cagr(equity_curve, years):
    if len(equity_curve) == 0:
        return np.nan

    final_value = equity_curve.iloc[-1]

    if final_value <= 0 or years <= 0:
        return np.nan

    return (final_value ** (1 / years)) - 1


def count_trades(signal):
    signal_changes = signal.diff().fillna(0) != 0
    entries = signal_changes & (signal != 0)
    return entries.sum()


def infer_periods_per_year(index, fallback=TRADING_DAYS_PER_YEAR):
    if len(index) < 2:
        return fallback

    if not hasattr(index, "dayofweek"):
        return fallback

    deltas = index.to_series().diff().dropna()

    if deltas.empty:
        return fallback

    median_days = deltas.median().total_seconds() / (60 * 60 * 24)

    if median_days <= 0:
        return fallback

    if median_days >= 2:
        return 365.25 / median_days

    has_weekends = (index.dayofweek >= 5).any()
    days_per_year = 365 if has_weekends else TRADING_DAYS_PER_YEAR

    if median_days >= 1:
        return days_per_year

    bars_per_day = index.to_series().groupby(index.normalize()).size().median()

    return bars_per_day * days_per_year


def summarize_strategy_results(results, periods_per_year=None):
    if periods_per_year is None:
        periods_per_year = infer_periods_per_year(results.index)

    years = calculate_act_act_years(results.index)

    return {
        "buy_hold_return": results["cum_asset"].iloc[-1] - 1,
        "strategy_return": results["cum_strategy"].iloc[-1] - 1,
        "max_drawdown": calculate_max_drawdown(results["cum_strategy"]),
        "longest_drawdown": calculate_longest_drawdown(results["cum_strategy"]),
        "sharpe_ratio": calculate_sharpe_ratio(results["strategy_return"], periods_per_year),
        "annualized_volatility": calculate_annualized_volatility(
            results["strategy_return"],
            periods_per_year,
        ),
        "mean": results["strategy_return"].mean(),
        "std": results["strategy_return"].std(),
        "cagr": calculate_cagr(results["cum_strategy"], years),
        "number_of_trades": count_trades(results["signal_raw"]),
        "periods_per_year": periods_per_year,
        "act_act_years": years,
    }

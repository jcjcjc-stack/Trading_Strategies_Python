from __future__ import annotations

import json
from collections.abc import Generator
from datetime import date, datetime
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

from analytics.performance_metrics import summarize_strategy_results


DEFAULT_TUNED_PARAMETERS_FILE = Path("research/optimization/tuned_hyperparameters.txt")
DEFAULT_TUNING_METADATA_FILE = Path("research/optimization/tuned_hyperparameters_metadata.txt")


def iter_parameter_grid(parameter_grid):
    if not parameter_grid:
        yield {}
        return

    keys = list(parameter_grid)
    values = [parameter_grid[key] for key in keys]

    for combination in product(*values):
        yield dict(zip(keys, combination))


def rolling_split(
    data: pd.DataFrame,
    train_size: float = 0.6,
    validation_size: float = 0.2,
) -> Generator[dict[str, object], None, None]:
    """
    Yield rolling train/validation windows from a time-indexed DataFrame.

    Defaults use 60% of the full dataset for training and 20% for validation.
    The final 20% is kept as test data outside this splitter.
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("rolling_split requires a pandas DataFrame with a DateTimeIndex.")

    if data.empty:
        raise ValueError("rolling_split requires at least one row.")

    if not data.index.is_monotonic_increasing:
        raise ValueError("DateTimeIndex must be sorted in ascending order.")

    if data.index.has_duplicates:
        raise ValueError("DateTimeIndex must not contain duplicate timestamps.")

    sizes = {"train_size": train_size, "validation_size": validation_size}
    for name, value in sizes.items():
        if not 0 < value < 1:
            raise ValueError(f"{name} must be between 0 and 1.")

    total = len(data)
    train_rows = int(total * train_size)
    validation_rows = int(total * validation_size)

    if train_rows == 0 or validation_rows == 0:
        raise ValueError("Split sizes are too small for the number of rows in data.")

    advance_rows = validation_rows
    fold = 1
    train_start = 0

    while True:
        train_end = train_start + train_rows
        validation_end = train_end + validation_rows

        if validation_end > total:
            break

        train = data.iloc[train_start:train_end].copy()
        validation = data.iloc[train_end:validation_end].copy()

        yield {
            "fold": fold,
            "train": train,
            "validation": validation,
            "train_start": train.index[0],
            "train_end": train.index[-1],
            "validation_start": validation.index[0],
            "validation_end": validation.index[-1],
        }

        fold += 1
        train_start += advance_rows


def metric_score(metrics, objective="sharpe_ratio"):
    value = metrics.get(objective)

    if value is None:
        return -np.inf

    try:
        if not np.isfinite(value):
            return -np.inf
    except TypeError:
        return -np.inf

    return value


def reset_equity_curves(results):
    results = results.copy()
    results["cum_asset"] = (1 + results["return"]).cumprod()
    results["cum_strategy"] = (1 + results["strategy_return"]).cumprod()
    return results


def tune_parameters(
    train,
    backtest_function,
    parameter_grid,
    periods_per_year=None,
    objective="sharpe_ratio",
):
    best = None

    for params in iter_parameter_grid(parameter_grid):
        results = backtest_function(train, **params)
        metrics = summarize_strategy_results(results, periods_per_year=periods_per_year)
        score = metric_score(metrics, objective=objective)

        if best is None or score > best["score"]:
            best = {
                "params": params,
                "metrics": metrics,
                "score": score,
            }

    return best


def rolling_validate(
    data,
    backtest_function,
    parameter_grid,
    train_size=0.6,
    validation_size=0.2,
    test_size=0.2,
    periods_per_year=None,
    objective="sharpe_ratio",
):
    total_size = train_size + validation_size + test_size
    if not np.isclose(total_size, 1.0):
        raise ValueError("train_size, validation_size, and test_size must add up to 1.")

    test_rows = int(len(data) * test_size)
    if test_rows == 0:
        raise ValueError("test_size is too small for the number of rows in data.")

    rolling_data = data.iloc[:-test_rows].copy()
    final_test_data = data.iloc[-test_rows:].copy()
    rolling_total = train_size + validation_size
    rolling_train_size = train_size / rolling_total
    rolling_validation_size = validation_size / rolling_total

    fold_results = []
    validation_results = []

    for split in rolling_split(
        rolling_data,
        train_size=rolling_train_size,
        validation_size=rolling_validation_size,
    ):
        tuned = tune_parameters(
            split["train"],
            backtest_function,
            parameter_grid,
            periods_per_year=periods_per_year,
            objective=objective,
        )

        evaluation_data = pd.concat([split["train"], split["validation"]])
        evaluated = backtest_function(evaluation_data, **tuned["params"])
        validation = reset_equity_curves(evaluated.loc[split["validation"].index])
        validation_metrics = summarize_strategy_results(
            validation,
            periods_per_year=periods_per_year,
        )

        fold_results.append(
            {
                **split,
                "best_params": tuned["params"],
                "train_metrics": tuned["metrics"],
                "train_score": tuned["score"],
                "validation_metrics": validation_metrics,
            }
        )
        validation_results.append(validation)

    if not validation_results:
        raise ValueError("Rolling validation did not produce any train/validation folds.")

    combined_validation = pd.concat(validation_results).sort_index()
    combined_validation = combined_validation[
        ~combined_validation.index.duplicated(keep="first")
    ].copy()
    combined_validation = reset_equity_curves(combined_validation)

    metrics = summarize_strategy_results(combined_validation, periods_per_year=periods_per_year)
    metrics["folds"] = len(fold_results)
    metrics["best_params_by_fold"] = [fold["best_params"] for fold in fold_results]
    metrics["avg_train_score"] = float(np.mean([fold["train_score"] for fold in fold_results]))

    return {
        "metrics": metrics,
        "folds": fold_results,
        "combined_validation": combined_validation,
        "final_test_data": final_test_data,
    }


def freeze_params(params):
    return tuple(sorted(params.items()))


def to_jsonable(value):
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, (date, datetime)):
        return value.isoformat()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}

    if isinstance(value, list):
        return [to_jsonable(item) for item in value]

    return value


def select_final_parameters(folds):
    """
    Pick one tuned parameter set from all rolling validation folds.

    The selected set is the one that appears most often across folds. Ties are
    resolved by average validation return, then average train score.
    """
    candidates = {}

    for fold in folds:
        key = freeze_params(fold["best_params"])

        if key not in candidates:
            candidates[key] = {
                "params": fold["best_params"],
                "count": 0,
                "validation_returns": [],
                "train_scores": [],
            }

        candidates[key]["count"] += 1
        candidates[key]["validation_returns"].append(
            fold["validation_metrics"]["strategy_return"]
        )
        candidates[key]["train_scores"].append(fold["train_score"])

    if not candidates:
        raise ValueError("Cannot select final parameters without rolling validation folds.")

    best = max(
        candidates.values(),
        key=lambda item: (
            item["count"],
            np.nanmean(item["validation_returns"]),
            np.nanmean(item["train_scores"]),
        ),
    )

    return dict(best["params"])


def save_tuned_parameters(tuned_parameters, output_path=DEFAULT_TUNED_PARAMETERS_FILE):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    clean_parameters = to_jsonable(tuned_parameters)
    output_path.write_text(
        json.dumps(clean_parameters, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return output_path


def metadata_path_for_parameters(output_path=DEFAULT_TUNED_PARAMETERS_FILE):
    output_path = Path(output_path)

    if output_path == DEFAULT_TUNED_PARAMETERS_FILE:
        return DEFAULT_TUNING_METADATA_FILE

    suffix = output_path.suffix or ".txt"
    return output_path.with_name(f"{output_path.stem}_metadata{suffix}")


def save_tuning_metadata(metadata, output_path=DEFAULT_TUNED_PARAMETERS_FILE):
    metadata_path = metadata_path_for_parameters(output_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    clean_metadata = to_jsonable(metadata)
    metadata_path.write_text(
        json.dumps(clean_metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return metadata_path


def load_tuned_parameters(input_path=DEFAULT_TUNED_PARAMETERS_FILE):
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Tuned parameters file does not exist: {input_path}")

    return json.loads(input_path.read_text(encoding="utf-8"))


def load_tuning_metadata(input_path=DEFAULT_TUNED_PARAMETERS_FILE):
    metadata_path = metadata_path_for_parameters(input_path)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Tuning metadata file does not exist: {metadata_path}")

    return json.loads(metadata_path.read_text(encoding="utf-8"))


def tune_all_strategy_parameters(
    data,
    strategies,
    parameter_grids,
    output_path=DEFAULT_TUNED_PARAMETERS_FILE,
    train_size=0.6,
    validation_size=0.2,
    test_size=0.2,
    periods_per_year=None,
    objective="sharpe_ratio",
):
    """
    Tune every strategy and write only final hyperparameters to a text file.

    strategies should be a dict of strategy name to backtest function.
    parameter_grids should be a dict of strategy name to parameter grid.
    """
    tuned_parameters = {}

    for strategy_name, backtest_function in strategies.items():
        validation = rolling_validate(
            data,
            backtest_function,
            parameter_grids[strategy_name],
            train_size=train_size,
            validation_size=validation_size,
            test_size=test_size,
            periods_per_year=periods_per_year,
            objective=objective,
        )
        tuned_parameters[strategy_name] = select_final_parameters(validation["folds"])

    save_tuned_parameters(tuned_parameters, output_path=output_path)
    return tuned_parameters


def get_tuned_parameters(strategy_name, input_path=DEFAULT_TUNED_PARAMETERS_FILE):
    tuned_parameters = load_tuned_parameters(input_path)
    return tuned_parameters.get(strategy_name, {})


def run_with_tuned_parameters(
    strategy_name,
    backtest_function,
    asset,
    input_path=DEFAULT_TUNED_PARAMETERS_FILE,
):
    params = get_tuned_parameters(strategy_name, input_path=input_path)
    return backtest_function(asset, **params)

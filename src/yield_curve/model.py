"""Recession probability model: dataset, time-based split, training and evaluation.

The model is a one-feature logistic regression: the monthly 10Y-3M spread at month
``t`` predicts whether a recession month occurs in ``(t, t + horizon]``.
"""

import pandas as pd

from yield_curve.data import DateLike, _to_timestamp
from yield_curve.features import (
    _validate_positive_int,
    monthly_spread,
    recession_within_horizon,
)

FEATURE = "spread"
TARGET = "target"
DEFAULT_HORIZON = 12
# Last month of information available for training (see time_split).
DEFAULT_CUTOFF = "2006-12"


def build_dataset(
    daily_spread: pd.Series, usrec: pd.Series, horizon: int = DEFAULT_HORIZON
) -> pd.DataFrame:
    """Build the modelling dataset: one row per month with a known target.

    Args:
        daily_spread: Daily 10Y-3M spread (``T10Y3M``), indexed by date.
        usrec: Monthly NBER recession indicator (``USREC``), indexed by month
            starts, with values 0, 1 or ``NaN``.
        horizon: Number of months ahead the target looks at. Defaults to 12.

    Returns:
        DataFrame indexed by month start (``date``) with two float columns:
        ``spread`` (monthly average spread at month ``t``) and ``target`` (1.0
        if a recession month occurs in ``(t, t + horizon]``, else 0.0). Months
        without a spread or with an unknown target (the last ``horizon`` months
        of ``usrec``) are dropped, so the result may be empty.

    Raises:
        TypeError: If an input has the wrong type (see ``monthly_spread`` and
            ``recession_within_horizon``).
        ValueError: If an input is malformed or ``horizon`` is below 1.
    """
    spread = monthly_spread(daily_spread)
    target = recession_within_horizon(usrec, horizon)
    dataset = pd.concat({FEATURE: spread, TARGET: target}, axis=1, join="inner")
    dataset = dataset.dropna().astype("float64")
    dataset.index.name = "date"
    return dataset


def _validate_dataset(dataset: object, name: str = "dataset") -> pd.DataFrame:
    """Check that ``dataset`` looks like the output of ``build_dataset``."""
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError(
            f"{name} must be a pandas DataFrame, got {type(dataset).__name__}."
        )
    if not isinstance(dataset.index, pd.DatetimeIndex):
        raise TypeError(
            f"{name} must be indexed by a DatetimeIndex, "
            f"got {type(dataset.index).__name__}."
        )
    missing = [column for column in (FEATURE, TARGET) if column not in dataset]
    if missing:
        raise ValueError(f"{name} is missing the columns {missing}.")
    if dataset.index.has_duplicates or not dataset.index.is_monotonic_increasing:
        raise ValueError(f"{name} must be sorted by date without duplicates.")
    if dataset[[FEATURE, TARGET]].isna().any().any():
        raise ValueError(f"{name} contains missing values; see build_dataset.")
    if not dataset[TARGET].isin([0, 1]).all():
        raise ValueError(f"{name} target must contain only 0 and 1.")
    return dataset


def _check_both_classes(frame: pd.DataFrame, name: str) -> None:
    """Raise if ``frame`` is empty or its target has a single class."""
    if frame.empty:
        raise ValueError(f"The {name} set is empty: move the cutoff date.")
    if frame[TARGET].nunique() < 2:
        raise ValueError(
            f"The {name} set has only one class (all targets are "
            f"{frame[TARGET].iloc[0]:.0f}): it needs recession and non-recession "
            "months. Move the cutoff date."
        )


def time_split(
    dataset: pd.DataFrame,
    cutoff: DateLike = DEFAULT_CUTOFF,
    horizon: int = DEFAULT_HORIZON,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset in time around a cutoff month, with an embargo.

    The cutoff month is the last month of information the model may use. The test
    set starts the month after the cutoff. The training set only keeps months
    whose whole target window ``(t, t + horizon]`` ends by the cutoff: the
    ``horizon`` months just before (and including) the cutoff are left out of both
    sets. This **embargo** prevents look-ahead: their targets depend on recession
    data from after the cutoff, which would not have been known at the time.

    With the defaults (cutoff 2006-12, horizon 12), training runs up to 2005-12
    and testing starts in 2007-01. Rows are never shuffled.

    Args:
        dataset: Output of ``build_dataset``.
        cutoff: Any date within the cutoff month. Defaults to ``"2006-12"``.
        horizon: Target horizon in months, which sets the embargo length. Must
            match the horizon used to build ``dataset``. Defaults to 12.

    Returns:
        A tuple ``(train, test)`` of DataFrames with the same columns as
        ``dataset``, in chronological order.

    Raises:
        TypeError: If ``dataset`` is not a DataFrame indexed by dates, ``cutoff``
            has an unsupported type, or ``horizon`` is not an integer.
        ValueError: If ``dataset`` is malformed, ``cutoff`` is not a valid date,
            ``horizon`` is below 1, or either set is empty or has one class only.
    """
    dataset = _validate_dataset(dataset)
    horizon = _validate_positive_int(horizon, "horizon")
    if cutoff is None:
        raise TypeError("cutoff must be a date, got None.")
    cutoff_month = _to_timestamp(cutoff, "cutoff").to_period("M").to_timestamp()

    last_train_month = cutoff_month - pd.offsets.MonthBegin(horizon)
    first_test_month = cutoff_month + pd.offsets.MonthBegin(1)
    train = dataset.loc[dataset.index <= last_train_month].copy()
    test = dataset.loc[dataset.index >= first_test_month].copy()

    _check_both_classes(train, "training")
    _check_both_classes(test, "test")
    return train, test

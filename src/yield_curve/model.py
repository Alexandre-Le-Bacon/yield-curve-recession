"""Recession probability model: dataset, time-based split, training and evaluation.

The model is a one-feature logistic regression: the monthly 10Y-3M spread at month
``t`` predicts whether a recession month occurs in ``(t, t + horizon]``.
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.utils.validation import check_is_fitted

from yield_curve.data import DateLike, _to_timestamp
from yield_curve.features import (
    _validate_positive_int,
    last_complete_month,
    monthly_spread,
    recession_within_horizon,
)

FEATURE = "spread"
TARGET = "target"
DEFAULT_HORIZON = 12
# Last month of information available for training (see time_split).
DEFAULT_CUTOFF = "2006-12"
RANDOM_STATE = 0


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


def fit_model(train: pd.DataFrame) -> LogisticRegression:
    """Fit a logistic regression of the target on the spread.

    Args:
        train: Training rows, e.g. the first output of ``time_split``.

    Returns:
        A fitted ``LogisticRegression`` with a fixed ``random_state``, so the
        result is the same on every run.

    Raises:
        TypeError: If ``train`` is not a DataFrame indexed by dates.
        ValueError: If ``train`` is malformed, empty, or has one class only.
    """
    train = _validate_dataset(train, "train")
    _check_both_classes(train, "training")
    model = LogisticRegression(random_state=RANDOM_STATE)
    model.fit(train[[FEATURE]], train[TARGET].astype(int))
    return model


def predict_probability(model: LogisticRegression, spread: pd.Series) -> pd.Series:
    """Predict the probability of a recession within the horizon for each month.

    Args:
        model: Model returned by ``fit_model``.
        spread: Monthly spread values, e.g. ``dataset["spread"]`` or the output of
            ``monthly_spread``.

    Returns:
        Float series with the same index as ``spread``, named ``"probability"``,
        with values in [0, 1]. Months with a missing spread get ``NaN``.

    Raises:
        TypeError: If ``model`` is not a ``LogisticRegression`` or ``spread`` is
            not a Series.
        ValueError: If ``model`` has not been fitted.
    """
    if not isinstance(model, LogisticRegression):
        raise TypeError(
            f"model must be a LogisticRegression, got {type(model).__name__}."
        )
    try:
        check_is_fitted(model)
    except NotFittedError as error:
        raise ValueError("model has not been fitted; use fit_model.") from error
    if not isinstance(spread, pd.Series):
        raise TypeError(f"spread must be a pandas Series, got {type(spread).__name__}.")

    probability = pd.Series(float("nan"), index=spread.index, name="probability")
    valid = spread.dropna()
    if not valid.empty:
        features = valid.astype("float64").to_frame(FEATURE)
        probability.loc[valid.index] = model.predict_proba(features)[:, 1]
    return probability


@dataclass(frozen=True)
class Evaluation:
    """Out-of-sample scores of the model and of the naive baseline.

    Attributes:
        model_auc: ROC AUC of the model on the test set.
        model_brier: Brier score of the model on the test set.
        baseline_auc: ROC AUC of the baseline (always 0.5: a constant prediction
            cannot rank months).
        baseline_brier: Brier score of the baseline on the test set.
        base_rate: Share of positive targets in the training set, used as the
            baseline prediction.
        train_start: First training month.
        train_end: Last training month.
        test_start: First test month.
        test_end: Last test month.
        n_train: Number of training months.
        n_test: Number of test months.
        n_test_positive: Number of test months with a recession within the horizon.
    """

    model_auc: float
    model_brier: float
    baseline_auc: float
    baseline_brier: float
    base_rate: float
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int
    n_test_positive: int


def fit_evaluation_model(
    dataset: pd.DataFrame,
    cutoff: DateLike = DEFAULT_CUTOFF,
    horizon: int = DEFAULT_HORIZON,
) -> LogisticRegression:
    """Fit the model on the training side of ``time_split`` only.

    Args:
        dataset: Output of ``build_dataset``.
        cutoff: Cutoff month, see ``time_split``.
        horizon: Target horizon in months, see ``time_split``.

    Returns:
        The fitted model, which has never seen the test months.

    Raises:
        TypeError: See ``time_split``.
        ValueError: See ``time_split``.
    """
    train, _ = time_split(dataset, cutoff, horizon)
    return fit_model(train)


def evaluate(
    dataset: pd.DataFrame,
    cutoff: DateLike = DEFAULT_CUTOFF,
    horizon: int = DEFAULT_HORIZON,
) -> Evaluation:
    """Evaluate the model on the test period against a naive baseline.

    The model is fitted on the training set only (see ``time_split``) and scored
    on the test set. The baseline predicts the training base rate for every month.
    Two scores are reported:

    - ROC AUC: probability that a random positive month gets a higher score than
      a random negative month (0.5 = no skill, 1 = perfect ranking).
    - Brier score: mean squared error between predicted probabilities and actual
      outcomes (0 = perfect, lower is better).

    Args:
        dataset: Output of ``build_dataset``.
        cutoff: Cutoff month, see ``time_split``.
        horizon: Target horizon in months, see ``time_split``.

    Returns:
        An ``Evaluation`` with the scores and the split details.

    Raises:
        TypeError: See ``time_split``.
        ValueError: See ``time_split``.
    """
    train, test = time_split(dataset, cutoff, horizon)
    model = fit_model(train)
    actual = test[TARGET].astype(int)
    predicted = predict_probability(model, test[FEATURE])
    base_rate = float(train[TARGET].mean())
    baseline = pd.Series(base_rate, index=test.index)

    return Evaluation(
        model_auc=float(roc_auc_score(actual, predicted)),
        model_brier=float(brier_score_loss(actual, predicted)),
        baseline_auc=float(roc_auc_score(actual, baseline)),
        baseline_brier=float(brier_score_loss(actual, baseline)),
        base_rate=base_rate,
        train_start=train.index[0],
        train_end=train.index[-1],
        test_start=test.index[0],
        test_end=test.index[-1],
        n_train=len(train),
        n_test=len(test),
        n_test_positive=int(actual.sum()),
    )


@dataclass(frozen=True)
class MonthReading:
    """Model output for one month.

    Attributes:
        month: First day of the month.
        spread: Average 10Y-3M spread over the available days of the month.
        probability: Predicted probability of a recession within the horizon.
    """

    month: pd.Timestamp
    spread: float
    probability: float


@dataclass(frozen=True)
class CurrentReading:
    """Today's reading of the model, fitted on all labelled months.

    Attributes:
        latest_complete: Reading for the last complete month (the headline).
        in_progress: Reading for the current, incomplete month, or ``None`` when
            the last month of data is complete.
        data_until: Date of the last valid daily spread observation.
        trained_from: First month used for training.
        trained_until: Last month used for training (last month with a known
            target).
    """

    latest_complete: MonthReading
    in_progress: MonthReading | None
    data_until: pd.Timestamp
    trained_from: pd.Timestamp
    trained_until: pd.Timestamp


def current_reading(
    daily_spread: pd.Series, usrec: pd.Series, horizon: int = DEFAULT_HORIZON
) -> CurrentReading:
    """Estimate today's recession probability, separately from the evaluation.

    Unlike ``evaluate``, the model is refitted on **all** months with a known
    target, then applied to the most recent months, whose target is not known yet.
    The headline uses the last complete month (see ``last_complete_month``); the
    current month, if its data is partial, is reported separately.

    Args:
        daily_spread: Daily 10Y-3M spread (``T10Y3M``), indexed by date.
        usrec: Monthly NBER recession indicator (``USREC``), indexed by month
            starts, with values 0, 1 or ``NaN``.
        horizon: Number of months ahead the target looks at. Defaults to 12.

    Returns:
        A ``CurrentReading`` with the readings and the training period.

    Raises:
        TypeError: If an input has the wrong type.
        ValueError: If an input is malformed, the labelled data is empty or has
            one class only, or the last complete month has no spread value.
    """
    dataset = build_dataset(daily_spread, usrec, horizon)
    model = fit_model(dataset)

    monthly = monthly_spread(daily_spread).dropna()
    probability = predict_probability(model, monthly)
    complete_month = last_complete_month(daily_spread)
    if complete_month not in monthly.index:
        raise ValueError(
            f"No spread data for the last complete month ({complete_month:%Y-%m})."
        )

    def reading(month: pd.Timestamp) -> MonthReading:
        return MonthReading(
            month=month,
            spread=float(monthly.loc[month]),
            probability=float(probability.loc[month]),
        )

    latest_month = monthly.index[-1]
    return CurrentReading(
        latest_complete=reading(complete_month),
        in_progress=reading(latest_month) if latest_month > complete_month else None,
        data_until=daily_spread.dropna().index[-1],
        trained_from=dataset.index[0],
        trained_until=dataset.index[-1],
    )

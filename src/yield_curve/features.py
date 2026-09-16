"""Feature engineering: yield spread, curve inversion detection and recession target.

All functions are pure: they take pandas objects indexed by dates, never modify
their inputs, and return new objects.
"""

import pandas as pd

from yield_curve.data import DateLike, _to_timestamp

Period = tuple[pd.Timestamp, pd.Timestamp]

# Treasury yield series, from shortest to longest maturity, with a short label.
YIELD_CURVE_SERIES: dict[str, str] = {
    "DGS3MO": "3M",
    "DGS2": "2Y",
    "DGS5": "5Y",
    "DGS10": "10Y",
    "DGS30": "30Y",
}
# Maturity of each label in years, e.g. for the x-axis of a yield curve chart.
MATURITY_YEARS: dict[str, float] = {
    "3M": 0.25,
    "2Y": 2.0,
    "5Y": 5.0,
    "10Y": 10.0,
    "30Y": 30.0,
}


def _validate_dated_series(series: object, name: str) -> pd.Series:
    """Check that ``series`` is a numeric Series with a sorted, unique date index."""
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pandas Series, got {type(series).__name__}.")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError(
            f"{name} must be indexed by a DatetimeIndex, "
            f"got {type(series.index).__name__}."
        )
    if not (pd.api.types.is_numeric_dtype(series) or series.empty):
        raise TypeError(f"{name} must contain numbers, got dtype {series.dtype}.")
    if series.index.has_duplicates:
        raise ValueError(f"{name} has duplicate dates.")
    if not series.index.is_monotonic_increasing:
        raise ValueError(f"{name} must be sorted by date.")
    return series


def _validate_monthly_series(series: object, name: str) -> pd.Series:
    """Check that ``series`` is a dated series whose dates are month starts."""
    series = _validate_dated_series(series, name)
    index = series.index
    if not ((index.day == 1) & (index == index.normalize())).all():
        raise ValueError(
            f"{name} must be indexed by month starts (e.g. 2020-01-01), "
            "see monthly_spread."
        )
    return series


def _validate_positive_int(value: object, name: str) -> int:
    """Check that ``value`` is an integer >= 1 (booleans are rejected)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}.")
    return value


def _flag_to_periods(flag: pd.Series, min_months: int = 1) -> list[Period]:
    """Turn a monthly boolean flag into periods of consecutive ``True`` months.

    Months absent from the index count as ``False``, so a gap ends a period.
    Each period is ``(first day of first month, last day of last month)``.
    """
    if flag.empty:
        return []
    full_range = pd.date_range(flag.index.min(), flag.index.max(), freq="MS")
    full = flag.reindex(full_range, fill_value=False).astype(bool)
    run_ids = (full != full.shift()).cumsum()

    periods: list[Period] = []
    for _, run in full.groupby(run_ids):
        if run.iloc[0] and len(run) >= min_months:
            start = run.index[0]
            end = run.index[-1] + pd.offsets.MonthEnd(0)
            periods.append((start, end))
    return periods


def monthly_spread(daily: pd.Series) -> pd.Series:
    """Average a daily series (e.g. the 10Y-3M spread) over each calendar month.

    Args:
        daily: Numeric series indexed by a sorted ``DatetimeIndex`` without
            duplicates. Missing values (``NaN``) are ignored in the averages.

    Returns:
        Float series with one value per month, indexed by the first day of each
        month (same convention as ``USREC``) and named like the input. A month
        with no valid observation is ``NaN``. An empty input gives an empty series.

    Raises:
        TypeError: If ``daily`` is not a numeric Series indexed by dates.
        ValueError: If the dates are unsorted or duplicated.
    """
    daily = _validate_dated_series(daily, "daily")
    monthly = daily.astype("float64").resample("MS").mean()
    monthly.index.name = "date"
    return monthly


def inversion_flag(monthly: pd.Series) -> pd.Series:
    """Flag the months where the yield curve is inverted (spread below zero).

    Args:
        monthly: Monthly spread, as returned by ``monthly_spread``.

    Returns:
        Boolean series named ``"inverted"`` with the same index. A spread of
        exactly 0 or a missing value (``NaN``) is not an inversion.

    Raises:
        TypeError: If ``monthly`` is not a numeric Series indexed by dates.
        ValueError: If the dates are unsorted, duplicated or not month starts.
    """
    monthly = _validate_monthly_series(monthly, "monthly")
    return (monthly < 0).rename("inverted")


def inversion_episodes(monthly: pd.Series, min_months: int = 1) -> list[Period]:
    """Group consecutive inverted months into inversion episodes.

    Args:
        monthly: Monthly spread, as returned by ``monthly_spread``.
        min_months: Minimum number of consecutive inverted months for an episode
            to be kept. Use a value above 1 to ignore short blips.

    Returns:
        List of ``(start, end)`` timestamps in chronological order, where
        ``start`` is the first day of the first inverted month and ``end`` the
        last day of the last inverted month (both inclusive). A missing month
        (``NaN`` or absent from the index) ends an episode.

    Raises:
        TypeError: If ``monthly`` is not a numeric Series indexed by dates, or
            ``min_months`` is not an integer.
        ValueError: If the dates are unsorted, duplicated or not month starts,
            or ``min_months`` is below 1.
    """
    min_months = _validate_positive_int(min_months, "min_months")
    return _flag_to_periods(inversion_flag(monthly), min_months)


def _validate_recession_indicator(usrec: object) -> pd.Series:
    """Check that ``usrec`` is a monthly series containing only 0, 1 or NaN."""
    usrec = _validate_monthly_series(usrec, "usrec")
    values = usrec.dropna()
    if not values.isin([0, 1]).all():
        raise ValueError("usrec must contain only 0, 1 or missing values.")
    return usrec


def recession_periods(usrec: pd.Series) -> list[Period]:
    """Turn the monthly recession indicator into recession periods.

    Args:
        usrec: Monthly NBER recession indicator (``USREC``): 1 during a
            recession month, 0 otherwise, indexed by month starts.

    Returns:
        List of ``(start, end)`` timestamps in chronological order, where
        ``start`` is the first day of the first recession month and ``end`` the
        last day of the last one (both inclusive). Useful for chart shading.

    Raises:
        TypeError: If ``usrec`` is not a numeric Series indexed by dates.
        ValueError: If the dates are unsorted, duplicated or not month starts, or
            a value is not 0, 1 or missing.
    """
    usrec = _validate_recession_indicator(usrec)
    return _flag_to_periods(usrec == 1)


def recession_within_horizon(usrec: pd.Series, horizon: int = 12) -> pd.Series:
    """Build the prediction target: is there a recession in the next months?

    For each month ``t``, the target is 1 if any month in ``(t, t + horizon]``
    is a recession month, and 0 otherwise. Month ``t`` itself is excluded: the
    question is whether a recession is *coming*.

    To avoid look-ahead leakage, the target is ``NaN`` whenever the window is
    not fully known: for the last ``horizon`` months of the data, and for months
    whose window contains a missing value or a month absent from the index.

    Args:
        usrec: Monthly NBER recession indicator (``USREC``), indexed by month
            starts, with values 0, 1 or ``NaN``.
        horizon: Number of months to look ahead. Defaults to 12.

    Returns:
        Float series (0.0, 1.0 or ``NaN``) with the same index as ``usrec``,
        named ``"recession_within_<horizon>m"``.

    Raises:
        TypeError: If ``usrec`` is not a numeric Series indexed by dates, or
            ``horizon`` is not an integer.
        ValueError: If the dates are unsorted, duplicated or not month starts, a
            value is not 0, 1 or missing, or ``horizon`` is below 1.
    """
    usrec = _validate_recession_indicator(usrec)
    horizon = _validate_positive_int(horizon, "horizon")
    name = f"recession_within_{horizon}m"
    if usrec.empty:
        return pd.Series([], index=usrec.index, name=name, dtype="float64")

    full_range = pd.date_range(usrec.index.min(), usrec.index.max(), freq="MS")
    full = usrec.astype("float64").reindex(full_range)
    # Rolling on the reversed series gives, at t, the max over [t, t + horizon - 1];
    # shift(-1) then moves the window to (t, t + horizon]. min_periods=horizon turns
    # a window that is incomplete (end of data or missing month) into NaN.
    forward_max = full[::-1].rolling(horizon, min_periods=horizon).max()[::-1]
    target = forward_max.shift(-1)
    return target.reindex(usrec.index).rename(name)


def yield_curve_on(
    yields: pd.DataFrame, when: DateLike
) -> tuple[pd.Timestamp, pd.Series]:
    """Get the yield curve on a date, or on the closest previous available date.

    The function uses the most recent row on or before ``when`` that has at least
    one yield. Maturities missing on that row are left out (for example, the
    30-year bond was not issued between 2002 and 2006).

    Args:
        yields: DataFrame indexed by a sorted ``DatetimeIndex``, with the columns
            of ``YIELD_CURVE_SERIES`` (``DGS3MO``, ``DGS2``, ``DGS5``, ``DGS10``,
            ``DGS30``), in percent. Other columns are ignored.
        when: Requested date (string, date, datetime or Timestamp).

    Returns:
        A tuple ``(as_of, curve)``: ``as_of`` is the date actually used, and
        ``curve`` a float series of yields named ``"yield"``, indexed by
        maturity label (``"3M"``, ``"2Y"``, ...) from shortest to longest.

    Raises:
        TypeError: If ``yields`` is not a DataFrame indexed by dates, or ``when``
            has an unsupported type.
        ValueError: If a yield column is missing, the dates are unsorted or
            duplicated, ``when`` is not a valid date, or no yield is available on
            or before ``when``.
    """
    if not isinstance(yields, pd.DataFrame):
        raise TypeError(
            f"yields must be a pandas DataFrame, got {type(yields).__name__}."
        )
    if not isinstance(yields.index, pd.DatetimeIndex):
        raise TypeError(
            "yields must be indexed by a DatetimeIndex, "
            f"got {type(yields.index).__name__}."
        )
    missing = [column for column in YIELD_CURVE_SERIES if column not in yields]
    if missing:
        raise ValueError(f"yields is missing the columns {missing}.")
    if yields.index.has_duplicates:
        raise ValueError("yields has duplicate dates.")
    if not yields.index.is_monotonic_increasing:
        raise ValueError("yields must be sorted by date.")
    if when is None:
        raise TypeError("when must be a date, got None.")
    timestamp = _to_timestamp(when, "when")

    available = yields.loc[yields.index <= timestamp, list(YIELD_CURVE_SERIES)]
    available = available.dropna(how="all")
    if available.empty:
        raise ValueError(f"No yield data on or before {timestamp.date()}.")

    as_of = available.index[-1]
    curve = available.iloc[-1].rename(YIELD_CURVE_SERIES).dropna().astype("float64")
    curve.index.name = "maturity"
    return as_of, curve.rename("yield")

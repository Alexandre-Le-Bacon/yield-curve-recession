"""Feature engineering: yield spread, curve inversion detection and recession target.

All functions are pure: they take pandas objects indexed by dates, never modify
their inputs, and return new objects.
"""

from dataclasses import dataclass

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


def last_complete_month(daily: pd.Series) -> pd.Timestamp:
    """Find the last calendar month fully covered by a daily business-day series.

    A month counts as complete when the last valid observation falls on or after
    its last business day (Monday to Friday). Otherwise the month is still in
    progress and the previous month is returned. The rule is conservative: if the
    last business day of a month is a market holiday, that month only counts as
    complete once data for the next month arrives.

    Args:
        daily: Numeric series indexed by a sorted ``DatetimeIndex`` without
            duplicates, e.g. the daily 10Y-3M spread.

    Returns:
        First day of the last complete month.

    Raises:
        TypeError: If ``daily`` is not a numeric Series indexed by dates.
        ValueError: If the dates are unsorted or duplicated, or the series has no
            valid observation.
    """
    daily = _validate_dated_series(daily, "daily")
    valid = daily.dropna()
    if valid.empty:
        raise ValueError("daily has no valid observation.")
    last_day = valid.index[-1].normalize()
    month_start = last_day.to_period("M").to_timestamp()
    if last_day >= last_day + pd.offsets.BMonthEnd(0):
        return month_start
    return month_start - pd.offsets.MonthBegin(1)


@dataclass(frozen=True)
class RecessionAfter:
    """What happened in the months after a given month.

    Attributes:
        month: First day of the reference month.
        months: Length of the window looked at, in months.
        first_recession_month: First recession month in the window, or ``None``.
        months_later: Number of months between ``month`` and
            ``first_recession_month``, or ``None``.
        complete: Whether every month of the window is known. When it is
            ``False`` and no recession was found, the answer may still change.
    """

    month: pd.Timestamp
    months: int
    first_recession_month: pd.Timestamp | None
    months_later: int | None
    complete: bool


def recession_after(
    usrec: pd.Series, when: DateLike, months: int = 24
) -> RecessionAfter:
    """Check whether a recession started in the months following a given month.

    The window is ``(month of when, month of when + months]``: the reference
    month itself is excluded.

    Args:
        usrec: Monthly NBER recession indicator (``USREC``), indexed by month
            starts, with values 0, 1 or ``NaN``.
        when: Any date within the reference month.
        months: Length of the window, in months. Defaults to 24.

    Returns:
        A ``RecessionAfter`` with the first recession month found in the window
        (if any) and whether the whole window is covered by known data.

    Raises:
        TypeError: If ``usrec`` is not a numeric Series indexed by dates, ``when``
            has an unsupported type, or ``months`` is not an integer.
        ValueError: If the dates are unsorted, duplicated or not month starts, a
            value is not 0, 1 or missing, ``when`` is not a valid date, or
            ``months`` is below 1.
    """
    usrec = _validate_recession_indicator(usrec)
    months = _validate_positive_int(months, "months")
    if when is None:
        raise TypeError("when must be a date, got None.")
    month = _to_timestamp(when, "when").to_period("M").to_timestamp()

    window = pd.date_range(
        month + pd.offsets.MonthBegin(1), periods=months, freq="MS", name="date"
    )
    values = usrec.reindex(window)
    recession_months = values.index[values == 1]
    complete = bool(values.notna().all())
    if recession_months.empty:
        return RecessionAfter(month, months, None, None, complete)

    first = recession_months[0]
    months_later = (first.year - month.year) * 12 + first.month - month.month
    return RecessionAfter(month, months, first, months_later, complete)


@dataclass(frozen=True)
class TrackRecord:
    """How well yield curve inversions lined up with recessions.

    Attributes:
        within_months: Window used to link an inversion and a recession, in months.
        min_months: Minimum length of an inversion episode, in months.
        recessions: Recessions that started after the first month of spread data.
        preceded: Those of ``recessions`` with an inversion episode starting in the
            ``within_months`` months before the recession started.
        episodes: All inversion episodes of at least ``min_months`` months.
        followed: Episodes followed by a recession starting within ``within_months``
            months of the episode start.
        false_alarms: Episodes not followed by a recession, although the whole
            window is known.
        pending: Episodes not followed by a recession so far, whose window is not
            complete yet: too recent to judge.
    """

    within_months: int
    min_months: int
    recessions: tuple[Period, ...]
    preceded: tuple[Period, ...]
    episodes: tuple[Period, ...]
    followed: tuple[Period, ...]
    false_alarms: tuple[Period, ...]
    pending: tuple[Period, ...]


def signal_track_record(
    monthly: pd.Series,
    usrec: pd.Series,
    within_months: int = 24,
    min_months: int = 3,
) -> TrackRecord:
    """Summarize the inversion signal: recessions it preceded and its false alarms.

    Args:
        monthly: Monthly spread, as returned by ``monthly_spread``.
        usrec: Monthly NBER recession indicator (``USREC``), indexed by month
            starts, with values 0, 1 or ``NaN``.
        within_months: Maximum gap between the start of an inversion and the start
            of a recession for the two to be linked. Defaults to 24.
        min_months: Minimum length of an inversion episode, see
            ``inversion_episodes``. Defaults to 3.

    Returns:
        A ``TrackRecord``. Only recessions starting after the first month with a
        spread value are considered, so that the curve before them is known. With
        no spread data at all, every field is empty.

    Raises:
        TypeError: If an input has the wrong type or an integer argument is not an
            integer.
        ValueError: If an input is malformed or an integer argument is below 1.
    """
    within_months = _validate_positive_int(within_months, "within_months")
    episodes = inversion_episodes(monthly, min_months)
    all_recessions = recession_periods(usrec)
    valid_spread = monthly.dropna()
    if valid_spread.empty:
        return TrackRecord(within_months, min_months, (), (), (), (), (), ())

    first_month = valid_spread.index[0]
    recessions = [period for period in all_recessions if period[0] > first_month]
    preceded = [
        (start, end)
        for start, end in recessions
        if any(
            start - pd.offsets.MonthBegin(within_months) <= episode_start < start
            for episode_start, _ in episodes
        )
    ]

    followed, false_alarms, pending = [], [], []
    for episode in episodes:
        after = recession_after(usrec, episode[0], months=within_months)
        if after.first_recession_month is not None:
            followed.append(episode)
        elif after.complete:
            false_alarms.append(episode)
        else:
            pending.append(episode)

    return TrackRecord(
        within_months=within_months,
        min_months=min_months,
        recessions=tuple(recessions),
        preceded=tuple(preceded),
        episodes=tuple(episodes),
        followed=tuple(followed),
        false_alarms=tuple(false_alarms),
        pending=tuple(pending),
    )


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

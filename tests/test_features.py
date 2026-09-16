from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from yield_curve.features import (
    MATURITY_YEARS,
    YIELD_CURVE_SERIES,
    inversion_episodes,
    inversion_flag,
    monthly_spread,
    recession_periods,
    recession_within_horizon,
    yield_curve_on,
)


def months(start: str, values: list[float], name: str = "T10Y3M") -> pd.Series:
    """Monthly series starting at ``start`` (month-start dates)."""
    index = pd.date_range(start, periods=len(values), freq="MS", name="date")
    return pd.Series(values, index=index, name=name, dtype="float64")


def ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value)


# --- monthly_spread ------------------------------------------------------------


class TestMonthlySpread:
    def test_averages_each_month(self):
        index = pd.to_datetime(["2020-01-02", "2020-01-31", "2020-02-03"])
        daily = pd.Series([1.0, 2.0, -0.5], index=index, name="T10Y3M")

        result = monthly_spread(daily)

        expected = months("2020-01-01", [1.5, -0.5])
        pd.testing.assert_series_equal(result, expected, check_freq=False)

    def test_ignores_missing_values(self):
        index = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
        daily = pd.Series([1.0, np.nan, 3.0], index=index, name="T10Y3M")
        assert monthly_spread(daily).tolist() == [2.0]

    def test_month_without_data_is_nan(self):
        index = pd.to_datetime(["2020-01-15", "2020-03-15"])
        daily = pd.Series([1.0, 3.0], index=index, name="T10Y3M")

        result = monthly_spread(daily)

        assert list(result.index) == [
            ts("2020-01-01"),
            ts("2020-02-01"),
            ts("2020-03-01"),
        ]
        assert np.isnan(result.iloc[1])

    def test_integer_input_gives_floats(self, daily_series):
        result = monthly_spread(daily_series.astype("int64"))
        assert result.dtype == "float64"
        assert result.tolist() == [4.5]

    def test_empty_series_gives_empty_result(self):
        empty = pd.Series([], index=pd.DatetimeIndex([]), dtype="float64")
        assert monthly_spread(empty).empty

    def test_does_not_modify_input(self, daily_series):
        before = daily_series.copy()
        monthly_spread(daily_series)
        pd.testing.assert_series_equal(daily_series, before)

    def test_rejects_non_series(self):
        with pytest.raises(TypeError, match="pandas Series"):
            monthly_spread([1.0, 2.0])

    def test_rejects_non_datetime_index(self):
        with pytest.raises(TypeError, match="DatetimeIndex"):
            monthly_spread(pd.Series([1.0, 2.0]))

    def test_rejects_non_numeric_values(self):
        index = pd.date_range("2020-01-01", periods=2, name="date")
        with pytest.raises(TypeError, match="numbers"):
            monthly_spread(pd.Series(["a", "b"], index=index))

    def test_rejects_unsorted_dates(self):
        index = pd.to_datetime(["2020-01-02", "2020-01-01"])
        with pytest.raises(ValueError, match="sorted"):
            monthly_spread(pd.Series([1.0, 2.0], index=index))

    def test_rejects_duplicate_dates(self):
        index = pd.to_datetime(["2020-01-01", "2020-01-01"])
        with pytest.raises(ValueError, match="duplicate"):
            monthly_spread(pd.Series([1.0, 2.0], index=index))


# --- inversion_flag ------------------------------------------------------------


class TestInversionFlag:
    def test_flags_negative_spread_only(self):
        result = inversion_flag(months("2020-01-01", [0.5, -0.1, 0.0, -2.0]))

        assert result.name == "inverted"
        assert result.dtype == bool
        assert result.tolist() == [False, True, False, True]

    def test_missing_value_is_not_inverted(self):
        assert inversion_flag(months("2020-01-01", [np.nan])).tolist() == [False]

    def test_empty_series(self):
        empty = pd.Series([], index=pd.DatetimeIndex([]), dtype="float64")
        assert inversion_flag(empty).empty

    def test_rejects_daily_dates(self, daily_series):
        with pytest.raises(ValueError, match="month starts"):
            inversion_flag(daily_series)

    def test_rejects_month_start_with_time(self):
        index = pd.DatetimeIndex([pd.Timestamp("2020-01-01 12:00")])
        with pytest.raises(ValueError, match="month starts"):
            inversion_flag(pd.Series([1.0], index=index))

    def test_rejects_non_series(self):
        with pytest.raises(TypeError):
            inversion_flag(pd.DataFrame({"a": [1.0]}))


# --- inversion_episodes --------------------------------------------------------


class TestInversionEpisodes:
    def test_groups_consecutive_inverted_months(self):
        spread = months("2020-01-01", [1.0, -0.2, -0.3, 0.5, -0.1, 0.2])

        assert inversion_episodes(spread) == [
            (ts("2020-02-01"), ts("2020-03-31")),
            (ts("2020-05-01"), ts("2020-05-31")),
        ]

    def test_episode_at_both_edges_of_the_data(self):
        spread = months("2020-01-01", [-1.0, 1.0, -1.0, -1.0])

        assert inversion_episodes(spread) == [
            (ts("2020-01-01"), ts("2020-01-31")),
            (ts("2020-03-01"), ts("2020-04-30")),
        ]

    def test_end_handles_february_in_leap_year(self):
        spread = months("2024-02-01", [-1.0])
        assert inversion_episodes(spread) == [(ts("2024-02-01"), ts("2024-02-29"))]

    def test_no_inversion_gives_empty_list(self):
        assert inversion_episodes(months("2020-01-01", [1.0, 0.0, 2.0])) == []

    def test_empty_series_gives_empty_list(self):
        empty = pd.Series([], index=pd.DatetimeIndex([]), dtype="float64")
        assert inversion_episodes(empty) == []

    def test_nan_month_splits_episode(self):
        spread = months("2020-01-01", [-1.0, np.nan, -1.0])

        assert inversion_episodes(spread) == [
            (ts("2020-01-01"), ts("2020-01-31")),
            (ts("2020-03-01"), ts("2020-03-31")),
        ]

    def test_month_absent_from_index_splits_episode(self):
        index = pd.to_datetime(["2020-01-01", "2020-03-01"])
        spread = pd.Series([-1.0, -1.0], index=index)

        assert len(inversion_episodes(spread)) == 2

    @pytest.mark.parametrize(
        ("min_months", "expected_starts"),
        [
            (1, ["2020-01-01", "2020-03-01", "2020-07-01"]),
            (2, ["2020-03-01", "2020-07-01"]),
            (3, ["2020-07-01"]),
            (4, []),
        ],
    )
    def test_min_months_filters_short_episodes(self, min_months, expected_starts):
        # Episodes of 1, 2 and 3 months.
        spread = months("2020-01-01", [-1, 1, -1, -1, 1, 1, -1, -1, -1, 1])

        result = inversion_episodes(spread, min_months=min_months)

        assert [start for start, _ in result] == [ts(d) for d in expected_starts]

    def test_min_months_keeps_full_episode_bounds(self):
        spread = months("2020-01-01", [-1, -1, -1])
        assert inversion_episodes(spread, min_months=3) == [
            (ts("2020-01-01"), ts("2020-03-31"))
        ]

    @pytest.mark.parametrize("min_months", [0, -1])
    def test_rejects_min_months_below_one(self, min_months):
        with pytest.raises(ValueError, match="at least 1"):
            inversion_episodes(months("2020-01-01", [-1.0]), min_months=min_months)

    @pytest.mark.parametrize("min_months", [1.5, "3", True])
    def test_rejects_non_integer_min_months(self, min_months):
        with pytest.raises(TypeError, match="integer"):
            inversion_episodes(months("2020-01-01", [-1.0]), min_months=min_months)

    def test_rejects_daily_series(self, daily_series):
        with pytest.raises(ValueError, match="month starts"):
            inversion_episodes(daily_series)


# --- recession_periods ---------------------------------------------------------


class TestRecessionPeriods:
    def test_groups_recession_months(self):
        usrec = months("2020-01-01", [0, 1, 1, 0, 0, 1], name="USREC")

        assert recession_periods(usrec) == [
            (ts("2020-02-01"), ts("2020-03-31")),
            (ts("2020-06-01"), ts("2020-06-30")),
        ]

    def test_no_recession_gives_empty_list(self):
        assert recession_periods(months("2020-01-01", [0, 0], name="USREC")) == []

    def test_empty_series_gives_empty_list(self):
        empty = pd.Series([], index=pd.DatetimeIndex([]), dtype="float64")
        assert recession_periods(empty) == []

    def test_missing_value_splits_period(self):
        usrec = months("2020-01-01", [1, np.nan, 1], name="USREC")
        assert len(recession_periods(usrec)) == 2

    def test_integer_indicator_is_accepted(self):
        usrec = months("2020-01-01", [1, 1], name="USREC").astype("int64")
        assert recession_periods(usrec) == [(ts("2020-01-01"), ts("2020-02-29"))]

    @pytest.mark.parametrize("bad_value", [2.0, -1.0, 0.5])
    def test_rejects_values_other_than_zero_or_one(self, bad_value):
        usrec = months("2020-01-01", [0, bad_value], name="USREC")
        with pytest.raises(ValueError, match="0, 1"):
            recession_periods(usrec)

    def test_rejects_daily_series(self, daily_series):
        with pytest.raises(ValueError, match="month starts"):
            recession_periods(daily_series.clip(upper=1))

    def test_rejects_non_series(self):
        with pytest.raises(TypeError, match="pandas Series"):
            recession_periods([0, 1])


# --- recession_within_horizon --------------------------------------------------


class TestRecessionWithinHorizon:
    def test_looks_at_the_next_months_only(self):
        # Recession in month 4 (index 3) only, horizon of 2 months.
        usrec = months("2020-01-01", [0, 0, 0, 1, 0, 0, 0, 0], name="USREC")

        result = recession_within_horizon(usrec, horizon=2)

        expected = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, np.nan, np.nan]
        np.testing.assert_array_equal(result.to_numpy(), expected)
        assert result.name == "recession_within_2m"
        pd.testing.assert_index_equal(result.index, usrec.index)

    def test_default_horizon_boundaries(self):
        # One recession month at index 13: visible from t = 1 (t + 12) up to t = 12.
        values = [0.0] * 30
        values[13] = 1.0
        usrec = months("2000-01-01", values, name="USREC")

        result = recession_within_horizon(usrec)

        assert result.name == "recession_within_12m"
        assert result.iloc[0] == 0.0  # recession at t + 13: out of the window
        assert result.iloc[1] == 1.0  # recession at t + 12: in the window
        assert result.iloc[12] == 1.0  # recession at t + 1
        assert result.iloc[13] == 0.0  # current month does not count

    def test_last_horizon_months_are_nan(self):
        usrec = months("2000-01-01", [1.0] * 24, name="USREC")

        result = recession_within_horizon(usrec)

        assert result.iloc[:12].eq(1.0).all()
        assert result.iloc[-12:].isna().all()

    def test_series_shorter_than_horizon_is_all_nan(self):
        usrec = months("2000-01-01", [0, 1, 0], name="USREC")
        assert recession_within_horizon(usrec).isna().all()

    def test_missing_value_in_window_gives_nan(self):
        usrec = months("2020-01-01", [0, 0, np.nan, 0, 0, 0], name="USREC")

        result = recession_within_horizon(usrec, horizon=2)

        np.testing.assert_array_equal(
            result.to_numpy(), [np.nan, np.nan, 0.0, 0.0, np.nan, np.nan]
        )

    def test_month_absent_from_index_gives_nan(self):
        index = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-04-01", "2020-05-01"])
        usrec = pd.Series([0.0, 0.0, 0.0, 0.0], index=index)

        result = recession_within_horizon(usrec, horizon=1)

        np.testing.assert_array_equal(result.to_numpy(), [0.0, np.nan, 0.0, np.nan])
        pd.testing.assert_index_equal(result.index, usrec.index)

    def test_empty_series_gives_empty_float_series(self):
        empty = pd.Series([], index=pd.DatetimeIndex([]), dtype="float64")

        result = recession_within_horizon(empty)

        assert result.empty
        assert result.dtype == "float64"

    def test_does_not_modify_input(self):
        usrec = months("2020-01-01", [0, 1, 0], name="USREC")
        before = usrec.copy()
        recession_within_horizon(usrec, horizon=1)
        pd.testing.assert_series_equal(usrec, before)

    @pytest.mark.parametrize("horizon", [0, -12])
    def test_rejects_horizon_below_one(self, horizon):
        usrec = months("2020-01-01", [0, 1], name="USREC")
        with pytest.raises(ValueError, match="at least 1"):
            recession_within_horizon(usrec, horizon=horizon)

    @pytest.mark.parametrize("horizon", [12.0, "12", None])
    def test_rejects_non_integer_horizon(self, horizon):
        usrec = months("2020-01-01", [0, 1], name="USREC")
        with pytest.raises(TypeError, match="integer"):
            recession_within_horizon(usrec, horizon=horizon)

    def test_rejects_invalid_indicator_values(self):
        usrec = months("2020-01-01", [0, 3], name="USREC")
        with pytest.raises(ValueError, match="0, 1"):
            recession_within_horizon(usrec)


# --- yield_curve_on ------------------------------------------------------------


@pytest.fixture
def yields() -> pd.DataFrame:
    """Three business days of Treasury yields, with gaps."""
    index = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    index.name = "date"
    return pd.DataFrame(
        {
            "DGS3MO": [1.5, np.nan, 1.6],
            "DGS2": [1.6, np.nan, 1.7],
            "DGS5": [1.7, np.nan, 1.8],
            "DGS10": [1.9, np.nan, 2.0],
            "DGS30": [2.3, np.nan, np.nan],
            "T10Y3M": [0.4, np.nan, 0.4],
        },
        index=index,
    )


class TestYieldCurveOn:
    def test_constants_cover_every_maturity(self):
        assert list(YIELD_CURVE_SERIES.values()) == list(MATURITY_YEARS)

    def test_exact_date(self, yields):
        as_of, curve = yield_curve_on(yields, "2020-01-02")

        assert as_of == ts("2020-01-02")
        expected = pd.Series(
            [1.5, 1.6, 1.7, 1.9, 2.3],
            index=pd.Index(["3M", "2Y", "5Y", "10Y", "30Y"], name="maturity"),
            name="yield",
        )
        pd.testing.assert_series_equal(curve, expected)

    def test_uses_closest_previous_date_when_no_row(self, yields):
        # 2020-01-04 is a Saturday: not in the data.
        as_of, _ = yield_curve_on(yields, "2020-01-04")
        assert as_of == ts("2020-01-02")

    def test_skips_rows_with_only_missing_yields(self, yields):
        as_of, _ = yield_curve_on(yields, "2020-01-03")
        assert as_of == ts("2020-01-02")

    def test_drops_missing_maturities(self, yields):
        as_of, curve = yield_curve_on(yields, "2020-01-06")

        assert as_of == ts("2020-01-06")
        assert list(curve.index) == ["3M", "2Y", "5Y", "10Y"]

    def test_date_after_the_data_uses_last_row(self, yields):
        as_of, _ = yield_curve_on(yields, "2030-01-01")
        assert as_of == ts("2020-01-06")

    @pytest.mark.parametrize(
        "when",
        [date(2020, 1, 2), datetime(2020, 1, 2, 18, 0), pd.Timestamp("2020-01-02")],
    )
    def test_accepts_date_types(self, yields, when):
        as_of, _ = yield_curve_on(yields, when)
        assert as_of == ts("2020-01-02")

    def test_does_not_modify_input(self, yields):
        before = yields.copy()
        yield_curve_on(yields, "2020-01-06")
        pd.testing.assert_frame_equal(yields, before)

    def test_date_before_the_data_raises(self, yields):
        with pytest.raises(ValueError, match="No yield data"):
            yield_curve_on(yields, "2019-12-31")

    def test_empty_frame_raises(self, yields):
        with pytest.raises(ValueError, match="No yield data"):
            yield_curve_on(yields.iloc[0:0], "2020-01-02")

    def test_missing_column_raises(self, yields):
        with pytest.raises(ValueError, match="DGS30"):
            yield_curve_on(yields.drop(columns="DGS30"), "2020-01-02")

    def test_invalid_date_string_raises(self, yields):
        with pytest.raises(ValueError, match="not a valid date"):
            yield_curve_on(yields, "not-a-date")

    @pytest.mark.parametrize("when", [None, 20200102])
    def test_invalid_date_type_raises(self, yields, when):
        with pytest.raises(TypeError):
            yield_curve_on(yields, when)

    def test_rejects_non_dataframe(self, yields):
        with pytest.raises(TypeError, match="DataFrame"):
            yield_curve_on(yields["DGS10"], "2020-01-02")

    def test_rejects_non_datetime_index(self, yields):
        with pytest.raises(TypeError, match="DatetimeIndex"):
            yield_curve_on(yields.reset_index(drop=True), "2020-01-02")

    def test_rejects_unsorted_dates(self, yields):
        with pytest.raises(ValueError, match="sorted"):
            yield_curve_on(yields.iloc[::-1], "2020-01-02")

    def test_rejects_duplicate_dates(self, yields):
        duplicated = pd.concat([yields.iloc[[0]], yields.iloc[[0]]])
        with pytest.raises(ValueError, match="duplicate"):
            yield_curve_on(duplicated, "2020-01-02")

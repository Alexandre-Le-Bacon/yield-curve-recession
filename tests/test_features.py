import numpy as np
import pandas as pd
import pytest

from yield_curve.features import (
    inversion_episodes,
    inversion_flag,
    monthly_spread,
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

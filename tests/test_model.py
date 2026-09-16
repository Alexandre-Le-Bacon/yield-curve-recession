from datetime import date

import numpy as np
import pandas as pd
import pytest

from yield_curve.model import (
    DEFAULT_CUTOFF,
    DEFAULT_HORIZON,
    FEATURE,
    TARGET,
    build_dataset,
    time_split,
)

# Recession months in the synthetic data: two before the default cutoff, two after.
RECESSIONS = [
    ("1993-01", "1993-06"),
    ("1999-01", "1999-04"),
    ("2009-01", "2009-06"),
    ("2014-01", "2014-03"),
]


def ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value)


def make_usrec(start: str = "1990-01", end: str = "2016-12") -> pd.Series:
    index = pd.date_range(start, end, freq="MS", name="date")
    usrec = pd.Series(0.0, index=index, name="USREC")
    for first, last in RECESSIONS:
        usrec.loc[first:last] = 1.0
    return usrec


def make_daily_spread(usrec: pd.Series, end: str | None = None) -> pd.Series:
    """Business-day spread: inverted (below 0) in the 12 months before recessions."""
    rng = np.random.default_rng(0)
    monthly = pd.Series(2.0, index=usrec.index) + rng.normal(0, 0.5, len(usrec))
    for first, _ in RECESSIONS:
        start = ts(first) - pd.offsets.MonthBegin(12)
        monthly.loc[start : ts(first) - pd.offsets.MonthBegin(1)] -= 2.5
    days = pd.bdate_range(
        usrec.index[0], end or usrec.index[-1] + pd.offsets.MonthEnd(0)
    )
    daily = monthly.reindex(days.to_period("M").to_timestamp()).to_numpy()
    return pd.Series(daily, index=pd.DatetimeIndex(days, name="date"), name="T10Y3M")


@pytest.fixture
def usrec() -> pd.Series:
    return make_usrec()


@pytest.fixture
def daily_spread(usrec) -> pd.Series:
    return make_daily_spread(usrec)


@pytest.fixture
def dataset(daily_spread, usrec) -> pd.DataFrame:
    return build_dataset(daily_spread, usrec)


# --- build_dataset -------------------------------------------------------------


class TestBuildDataset:
    def test_columns_and_index(self, dataset, usrec):
        assert list(dataset.columns) == [FEATURE, TARGET]
        assert (dataset.dtypes == "float64").all()
        assert dataset.index.name == "date"
        assert dataset.index[0] == usrec.index[0]

    def test_drops_months_with_unknown_target(self, dataset, usrec):
        # The last 12 months of USREC have no known target.
        assert dataset.index[-1] == usrec.index[-1] - pd.offsets.MonthBegin(12)
        assert not dataset.isna().any().any()

    def test_feature_is_the_monthly_average(self, dataset, daily_spread):
        january = daily_spread.loc["1990-01"].mean()
        assert dataset.loc["1990-01-01", FEATURE] == pytest.approx(january)

    def test_target_matches_recession_within_horizon(self, dataset):
        # 1993-01 recession: 1992-01 is 12 months before, 1991-12 is 13.
        assert dataset.loc["1992-01-01", TARGET] == 1.0
        assert dataset.loc["1991-12-01", TARGET] == 0.0

    def test_horizon_changes_target_and_rows(self, daily_spread, usrec):
        short = build_dataset(daily_spread, usrec, horizon=6)
        assert short.loc["1992-01-01", TARGET] == 0.0
        assert short.index[-1] == usrec.index[-1] - pd.offsets.MonthBegin(6)

    def test_only_months_present_in_both_inputs(self, daily_spread, usrec):
        dataset = build_dataset(daily_spread.loc["1995":], usrec)
        assert dataset.index[0] == ts("1995-01-01")

    def test_months_without_spread_are_dropped(self, daily_spread, usrec):
        daily = daily_spread.copy()
        daily.loc["1995-03"] = np.nan
        assert ts("1995-03-01") not in build_dataset(daily, usrec).index

    def test_no_overlap_gives_empty_dataset(self, daily_spread, usrec):
        dataset = build_dataset(daily_spread.loc["2016"], usrec.loc[:"2010"])
        assert dataset.empty
        assert list(dataset.columns) == [FEATURE, TARGET]

    def test_does_not_modify_inputs(self, daily_spread, usrec):
        before = (daily_spread.copy(), usrec.copy())
        build_dataset(daily_spread, usrec)
        pd.testing.assert_series_equal(daily_spread, before[0])
        pd.testing.assert_series_equal(usrec, before[1])

    def test_rejects_invalid_horizon(self, daily_spread, usrec):
        with pytest.raises(ValueError, match="at least 1"):
            build_dataset(daily_spread, usrec, horizon=0)

    def test_rejects_invalid_usrec(self, daily_spread, usrec):
        with pytest.raises(ValueError, match="0, 1"):
            build_dataset(daily_spread, usrec * 3)

    def test_rejects_non_series(self, usrec):
        with pytest.raises(TypeError):
            build_dataset([1.0, 2.0], usrec)


# --- time_split ----------------------------------------------------------------


class TestTimeSplit:
    def test_default_split_with_embargo(self, dataset):
        assert DEFAULT_CUTOFF == "2006-12"
        assert DEFAULT_HORIZON == 12

        train, test = time_split(dataset)

        assert train.index[0] == dataset.index[0]
        assert train.index[-1] == ts("2005-12-01")
        assert test.index[0] == ts("2007-01-01")
        assert test.index[-1] == dataset.index[-1]

    def test_no_overlap_and_embargo_rows_in_neither_set(self, dataset):
        train, test = time_split(dataset)

        assert train.index.intersection(test.index).empty
        assert train.index.max() < test.index.min()
        embargoed = dataset.index.difference(train.index.union(test.index))
        assert list(embargoed) == list(pd.date_range("2006-01", "2006-12", freq="MS"))

    def test_training_targets_only_use_data_up_to_the_cutoff(self, dataset):
        train, _ = time_split(dataset)
        last_window_end = train.index[-1] + pd.offsets.MonthBegin(DEFAULT_HORIZON)
        assert last_window_end == ts("2006-12-01")

    @pytest.mark.parametrize("horizon", [1, 6, 12, 24])
    def test_embargo_length_follows_the_horizon(self, daily_spread, usrec, horizon):
        dataset = build_dataset(daily_spread, usrec, horizon=horizon)

        train, test = time_split(dataset, cutoff="2006-12", horizon=horizon)

        assert train.index[-1] == ts("2006-12-01") - pd.offsets.MonthBegin(horizon)
        assert test.index[0] == ts("2007-01-01")
        embargoed = dataset.index[
            (dataset.index > train.index[-1]) & (dataset.index < test.index[0])
        ]
        assert len(embargoed) == horizon

    @pytest.mark.parametrize(
        "cutoff", ["2006-12", "2006-12-31", date(2006, 12, 15), ts("2006-12-01")]
    )
    def test_any_date_in_the_cutoff_month(self, dataset, cutoff):
        train, test = time_split(dataset, cutoff=cutoff)
        assert train.index[-1] == ts("2005-12-01")
        assert test.index[0] == ts("2007-01-01")

    def test_rows_stay_in_chronological_order(self, dataset):
        train, test = time_split(dataset)
        assert train.index.is_monotonic_increasing
        assert test.index.is_monotonic_increasing
        pd.testing.assert_frame_equal(train, dataset.loc[:"2005-12"])

    def test_does_not_modify_input(self, dataset):
        before = dataset.copy()
        time_split(dataset)
        pd.testing.assert_frame_equal(dataset, before)

    def test_cutoff_before_the_data_gives_empty_training_set(self, dataset):
        with pytest.raises(ValueError, match="training set is empty"):
            time_split(dataset, cutoff="1990-06")

    def test_cutoff_after_the_data_gives_empty_test_set(self, dataset):
        with pytest.raises(ValueError, match="test set is empty"):
            time_split(dataset, cutoff="2030-01")

    def test_single_class_training_set(self, dataset):
        # Training up to 1991-12: no recession-within-12-months target yet.
        with pytest.raises(ValueError, match="training set has only one class"):
            time_split(dataset, cutoff="1992-12")

    def test_single_class_test_set(self, dataset):
        # Test from 2015-01: the last recession (2014) is already over.
        with pytest.raises(ValueError, match="test set has only one class"):
            time_split(dataset, cutoff="2014-12")

    def test_rejects_non_dataframe(self, dataset):
        with pytest.raises(TypeError, match="DataFrame"):
            time_split(dataset[FEATURE])

    def test_rejects_non_datetime_index(self, dataset):
        with pytest.raises(TypeError, match="DatetimeIndex"):
            time_split(dataset.reset_index(drop=True))

    def test_rejects_missing_column(self, dataset):
        with pytest.raises(ValueError, match="missing the columns"):
            time_split(dataset.drop(columns=TARGET))

    def test_rejects_unsorted_dataset(self, dataset):
        with pytest.raises(ValueError, match="sorted"):
            time_split(dataset.iloc[::-1])

    def test_rejects_missing_values(self, dataset):
        broken = dataset.copy()
        broken.iloc[3, 0] = np.nan
        with pytest.raises(ValueError, match="missing values"):
            time_split(broken)

    def test_rejects_non_binary_target(self, dataset):
        broken = dataset.copy()
        broken.iloc[3, 1] = 0.5
        with pytest.raises(ValueError, match="only 0 and 1"):
            time_split(broken)

    @pytest.mark.parametrize("cutoff", [None, 2006])
    def test_rejects_invalid_cutoff_type(self, dataset, cutoff):
        with pytest.raises(TypeError):
            time_split(dataset, cutoff=cutoff)

    def test_rejects_invalid_cutoff_string(self, dataset):
        with pytest.raises(ValueError, match="not a valid date"):
            time_split(dataset, cutoff="end of 2006")

    @pytest.mark.parametrize(("horizon", "error"), [(0, ValueError), (1.5, TypeError)])
    def test_rejects_invalid_horizon(self, dataset, horizon, error):
        with pytest.raises(error):
            time_split(dataset, horizon=horizon)

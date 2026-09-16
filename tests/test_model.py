from datetime import date

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from yield_curve.model import (
    DEFAULT_CUTOFF,
    DEFAULT_HORIZON,
    FEATURE,
    TARGET,
    CurrentReading,
    Evaluation,
    build_dataset,
    current_reading,
    evaluate,
    fit_evaluation_model,
    fit_model,
    predict_probability,
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


# --- fit_model / predict_probability -------------------------------------------


class TestFitAndPredict:
    def test_is_deterministic(self, dataset):
        first = fit_model(dataset)
        second = fit_model(dataset)

        np.testing.assert_array_equal(first.coef_, second.coef_)
        np.testing.assert_array_equal(first.intercept_, second.intercept_)
        pd.testing.assert_series_equal(
            predict_probability(first, dataset[FEATURE]),
            predict_probability(second, dataset[FEATURE]),
        )

    def test_probabilities_are_between_0_and_1(self, dataset):
        model = fit_model(dataset)
        extreme = pd.Series([-50.0, -1.0, 0.0, 1.0, 50.0])

        probability = predict_probability(model, pd.concat([dataset[FEATURE], extreme]))

        assert probability.between(0, 1).all()
        assert probability.name == "probability"

    def test_lower_spread_gives_higher_probability(self, dataset):
        model = fit_model(dataset)
        spreads = pd.Series([3.0, 2.0, 1.0, 0.0, -1.0])

        probability = predict_probability(model, spreads)

        assert probability.is_monotonic_increasing
        assert probability.iloc[-1] > probability.iloc[0]
        assert model.coef_[0, 0] < 0

    def test_keeps_index_and_gives_nan_for_missing_spread(self, dataset):
        model = fit_model(dataset)
        spread = pd.Series([1.0, np.nan], index=pd.to_datetime(["2020-01", "2020-02"]))

        probability = predict_probability(model, spread)

        pd.testing.assert_index_equal(probability.index, spread.index)
        assert 0 <= probability.iloc[0] <= 1
        assert np.isnan(probability.iloc[1])

    def test_empty_spread_gives_empty_result(self, dataset):
        model = fit_model(dataset)
        assert predict_probability(model, pd.Series([], dtype="float64")).empty

    def test_does_not_modify_training_data(self, dataset):
        before = dataset.copy()
        fit_model(dataset)
        pd.testing.assert_frame_equal(dataset, before)

    def test_fit_rejects_single_class(self, dataset):
        only_negative = dataset[dataset[TARGET] == 0]
        with pytest.raises(ValueError, match="only one class"):
            fit_model(only_negative)

    def test_fit_rejects_empty(self, dataset):
        with pytest.raises(ValueError, match="empty"):
            fit_model(dataset.iloc[0:0])

    def test_fit_rejects_non_dataframe(self, dataset):
        with pytest.raises(TypeError, match="DataFrame"):
            fit_model(dataset.to_numpy())

    def test_predict_rejects_other_model_types(self, dataset):
        with pytest.raises(TypeError, match="LogisticRegression"):
            predict_probability("model", dataset[FEATURE])

    def test_predict_rejects_unfitted_model(self, dataset):
        with pytest.raises(ValueError, match="not been fitted"):
            predict_probability(LogisticRegression(), dataset[FEATURE])

    def test_predict_rejects_non_series(self, dataset):
        with pytest.raises(TypeError, match="pandas Series"):
            predict_probability(fit_model(dataset), [1.0, 2.0])


# --- evaluate ------------------------------------------------------------------


class TestEvaluate:
    def test_reports_split_details(self, dataset):
        result = evaluate(dataset)

        assert isinstance(result, Evaluation)
        assert result.train_start == dataset.index[0]
        assert result.train_end == ts("2005-12-01")
        assert result.test_start == ts("2007-01-01")
        assert result.test_end == dataset.index[-1]
        assert result.n_train == len(dataset.loc[:"2005-12"])
        assert result.n_test == len(dataset.loc["2007-01":])
        assert result.n_test_positive == int(dataset.loc["2007-01":, TARGET].sum())

    def test_baseline_uses_training_base_rate(self, dataset):
        result = evaluate(dataset)

        train_rate = dataset.loc[:"2005-12", TARGET].mean()
        test_target = dataset.loc["2007-01":, TARGET]
        assert result.base_rate == pytest.approx(train_rate)
        assert result.baseline_auc == 0.5
        assert result.baseline_brier == pytest.approx(
            ((test_target - train_rate) ** 2).mean()
        )

    def test_scores_are_in_valid_ranges(self, dataset):
        result = evaluate(dataset)
        assert 0 <= result.model_auc <= 1
        assert 0 <= result.model_brier <= 1

    def test_model_beats_baseline_on_informative_data(self, dataset):
        result = evaluate(dataset)
        assert result.model_auc > 0.9
        assert result.model_brier < result.baseline_brier

    def test_is_deterministic(self, dataset):
        assert evaluate(dataset) == evaluate(dataset)

    def test_scores_only_depend_on_training_rows_for_fitting(self, dataset):
        # Changing test-period features must not change the fitted model.
        altered = dataset.copy()
        altered.loc["2007-01":, FEATURE] = 99.0
        original = fit_evaluation_model(dataset)
        refit = fit_evaluation_model(altered)
        np.testing.assert_array_equal(original.coef_, refit.coef_)

    def test_embargoed_rows_are_not_used_for_fitting(self, dataset):
        altered = dataset.copy()
        altered.loc["2006-01":"2006-12", FEATURE] = -99.0
        np.testing.assert_array_equal(
            fit_evaluation_model(dataset).coef_, fit_evaluation_model(altered).coef_
        )

    def test_custom_cutoff(self, dataset):
        result = evaluate(dataset, cutoff="2010-06")
        assert result.train_end == ts("2009-06-01")
        assert result.test_start == ts("2010-07-01")

    def test_result_is_immutable(self, dataset):
        result = evaluate(dataset)
        with pytest.raises(AttributeError):
            result.model_auc = 1.0

    def test_invalid_split_raises(self, dataset):
        with pytest.raises(ValueError, match="test set has only one class"):
            evaluate(dataset, cutoff="2014-12")

    def test_fit_evaluation_model_invalid_split_raises(self, dataset):
        with pytest.raises(ValueError, match="training set is empty"):
            fit_evaluation_model(dataset, cutoff="1990-01")


# --- current_reading -----------------------------------------------------------


class TestCurrentReading:
    @pytest.fixture
    def partial_spread(self, usrec) -> pd.Series:
        # Spread data runs until mid-February 2017, after the end of USREC.
        return make_daily_spread(make_usrec(end="2017-02"), end="2017-02-15")

    def test_headline_is_the_last_complete_month(self, partial_spread, usrec):
        result = current_reading(partial_spread, usrec)

        assert isinstance(result, CurrentReading)
        assert result.latest_complete.month == ts("2017-01-01")
        assert result.data_until == ts("2017-02-15")

    def test_in_progress_month_is_reported_separately(self, partial_spread, usrec):
        result = current_reading(partial_spread, usrec)

        assert result.in_progress is not None
        assert result.in_progress.month == ts("2017-02-01")
        february = partial_spread.loc["2017-02"].mean()
        assert result.in_progress.spread == pytest.approx(february)

    def test_no_in_progress_month_when_data_ends_on_month_end(
        self, daily_spread, usrec
    ):
        result = current_reading(daily_spread, usrec)

        assert result.latest_complete.month == ts("2016-12-01")
        assert result.in_progress is None

    def test_uses_a_model_fitted_on_all_labelled_months(self, partial_spread, usrec):
        result = current_reading(partial_spread, usrec)
        dataset = build_dataset(partial_spread, usrec)

        assert result.trained_from == dataset.index[0]
        assert result.trained_until == dataset.index[-1] == ts("2015-12-01")
        expected = predict_probability(
            fit_model(dataset), pd.Series([result.latest_complete.spread])
        ).iloc[0]
        assert result.latest_complete.probability == pytest.approx(expected)

    def test_reading_month_has_no_known_target(self, partial_spread, usrec):
        result = current_reading(partial_spread, usrec)
        assert result.latest_complete.month > result.trained_until

    def test_probabilities_are_between_0_and_1(self, partial_spread, usrec):
        result = current_reading(partial_spread, usrec)
        assert 0 <= result.latest_complete.probability <= 1
        assert 0 <= result.in_progress.probability <= 1

    def test_is_deterministic(self, partial_spread, usrec):
        assert current_reading(partial_spread, usrec) == current_reading(
            partial_spread, usrec
        )

    def test_missing_complete_month_raises(self, partial_spread, usrec):
        daily = partial_spread.copy()
        daily.loc["2017-01"] = np.nan
        # The last valid day is in February: January is complete but has no data.
        with pytest.raises(ValueError, match="last complete month"):
            current_reading(daily, usrec)

    def test_single_class_raises(self, partial_spread, usrec):
        with pytest.raises(ValueError, match="only one class"):
            current_reading(partial_spread, usrec * 0)

    def test_invalid_input_raises(self, usrec):
        with pytest.raises(TypeError):
            current_reading("T10Y3M", usrec)


# --- committed snapshots --------------------------------------------------------


def test_default_evaluation_and_reading_run_on_committed_snapshots():
    from yield_curve.data import load_series

    daily_spread, usrec = load_series("T10Y3M"), load_series("USREC")
    dataset = build_dataset(daily_spread, usrec)

    result = evaluate(dataset)
    reading = current_reading(daily_spread, usrec)

    assert result.train_end == ts("2005-12-01")
    assert result.test_start == ts("2007-01-01")
    assert result.n_test_positive > 0
    assert 0 <= reading.latest_complete.probability <= 1
    assert reading.latest_complete.month > reading.trained_until

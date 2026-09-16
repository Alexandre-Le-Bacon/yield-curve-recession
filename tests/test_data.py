import importlib
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from yield_curve import data as data_module
from yield_curve.data import (
    DATA_DIR_ENV_VAR,
    SERIES_IDS,
    filter_by_date,
    get_default_data_dir,
    load_download_date,
    load_series,
    load_series_frame,
    parse_fred_csv,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# --- get_default_data_dir / DEFAULT_DATA_DIR ------------------------------------


class TestDefaultDataDir:
    def test_falls_back_to_repo_data_raw(self):
        assert get_default_data_dir() == REPO_ROOT / "data" / "raw"

    def test_uses_env_var_when_set(self, monkeypatch, tmp_path):
        monkeypatch.setenv(DATA_DIR_ENV_VAR, str(tmp_path))
        assert get_default_data_dir() == tmp_path.resolve()

    def test_env_var_expands_home(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv(DATA_DIR_ENV_VAR, "~/snapshots")
        assert get_default_data_dir() == (tmp_path / "snapshots").resolve()

    @pytest.mark.parametrize("value", ["", "   "])
    def test_blank_env_var_is_ignored(self, monkeypatch, value):
        monkeypatch.setenv(DATA_DIR_ENV_VAR, value)
        assert get_default_data_dir() == REPO_ROOT / "data" / "raw"

    def test_falls_back_to_cwd_without_project_root(self, monkeypatch, tmp_path):
        # Simulate a non-editable install: no pyproject.toml above the module file.
        fake_module = tmp_path / "site-packages" / "yield_curve" / "data.py"
        monkeypatch.setattr(data_module, "__file__", str(fake_module))
        monkeypatch.chdir(tmp_path)
        assert get_default_data_dir() == (tmp_path / "data" / "raw").resolve()

    def test_constant_is_resolved_from_env_at_import(self, monkeypatch, tmp_path):
        monkeypatch.setenv(DATA_DIR_ENV_VAR, str(tmp_path))
        try:
            reloaded = importlib.reload(data_module)
            assert tmp_path.resolve() == reloaded.DEFAULT_DATA_DIR
        finally:
            monkeypatch.delenv(DATA_DIR_ENV_VAR)
            importlib.reload(data_module)

    def test_constant_defaults_to_repo_data_raw(self):
        assert REPO_ROOT / "data" / "raw" == data_module.DEFAULT_DATA_DIR


# --- parse_fred_csv -------------------------------------------------------------


class TestParseFredCsv:
    def test_parses_dates_and_values(self):
        csv = "observation_date,DGS10\n2020-01-02,1.88\n2020-01-03,1.80\n"
        series = parse_fred_csv(csv, "DGS10")

        assert series.name == "DGS10"
        assert series.dtype == "float64"
        assert isinstance(series.index, pd.DatetimeIndex)
        assert series.index.name == "date"
        assert series.to_dict() == {
            pd.Timestamp("2020-01-02"): 1.88,
            pd.Timestamp("2020-01-03"): 1.80,
        }

    def test_accepts_legacy_date_header(self):
        series = parse_fred_csv("DATE,USREC\n2020-01-01,1\n", "USREC")
        assert series.iloc[0] == 1.0

    @pytest.mark.parametrize("missing", [".", "", "  "])
    def test_missing_markers_become_nan(self, missing):
        csv = f"observation_date,DGS2\n2020-01-02,1.5\n2020-01-03,{missing}\n"
        series = parse_fred_csv(csv, "DGS2")
        assert len(series) == 2
        assert pd.isna(series.loc["2020-01-03"])

    def test_all_values_missing(self):
        series = parse_fred_csv("DATE,DGS5\n2020-01-01,.\n2020-01-02,\n", "DGS5")
        assert len(series) == 2
        assert series.isna().all()

    def test_sorts_by_date(self):
        csv = "DATE,DGS30\n2020-01-03,3\n2020-01-01,1\n2020-01-02,2\n"
        series = parse_fred_csv(csv, "DGS30")
        assert series.index.is_monotonic_increasing
        assert series.tolist() == [1.0, 2.0, 3.0]

    def test_strips_whitespace(self):
        series = parse_fred_csv("DATE,DGS2\n 2020-01-01 , 1.25 \n", "DGS2")
        assert series.loc["2020-01-01"] == 1.25

    def test_header_only_gives_empty_series(self):
        series = parse_fred_csv("observation_date,T10Y3M\n", "T10Y3M")
        assert series.empty
        assert series.dtype == "float64"
        assert isinstance(series.index, pd.DatetimeIndex)

    def test_negative_values(self):
        series = parse_fred_csv("DATE,T10Y3M\n2023-05-01,-1.7\n", "T10Y3M")
        assert series.iloc[0] == -1.7

    @pytest.mark.parametrize("text", ["", "   \n\n"])
    def test_empty_text_raises(self, text):
        with pytest.raises(ValueError, match="is empty"):
            parse_fred_csv(text, "T10Y3M")

    @pytest.mark.parametrize(
        "header",
        [
            "observation_date,DGS10",  # value column for another series
            "day,T10Y3M",  # unknown date column
            "T10Y3M,observation_date",  # columns swapped
            "observation_date,T10Y3M,extra",  # too many columns
            "observation_date",  # too few columns
        ],
    )
    def test_unexpected_columns_raise(self, header):
        with pytest.raises(ValueError, match="unexpected columns"):
            parse_fred_csv(f"{header}\n", "T10Y3M")

    @pytest.mark.parametrize("bad_date", ["2020-13-01", "01/02/2020", "yesterday"])
    def test_invalid_date_raises(self, bad_date):
        with pytest.raises(ValueError, match="invalid date"):
            parse_fred_csv(f"DATE,USREC\n{bad_date},1\n", "USREC")

    def test_missing_date_raises(self):
        with pytest.raises(ValueError, match="missing date"):
            parse_fred_csv("DATE,USREC\n2020-01-01,0\n,1\n", "USREC")

    def test_non_numeric_value_raises(self):
        with pytest.raises(ValueError, match="non-numeric"):
            parse_fred_csv("DATE,USREC\n2020-01-01,yes\n", "USREC")

    def test_duplicate_dates_raise(self):
        with pytest.raises(ValueError, match="duplicate dates"):
            parse_fred_csv("DATE,USREC\n2020-01-01,0\n2020-01-01,1\n", "USREC")

    def test_ragged_rows_raise(self):
        with pytest.raises(ValueError, match="CSV for USREC"):
            parse_fred_csv("DATE,USREC\n2020-01-01,0\n2020-02-01,1,2,3\n", "USREC")

    def test_unknown_series_id_raises(self):
        with pytest.raises(ValueError, match="Unknown series id 'GDP'"):
            parse_fred_csv("DATE,GDP\n2020-01-01,1\n", "GDP")

    def test_non_string_series_id_raises(self):
        with pytest.raises(TypeError, match="series_id must be a string"):
            parse_fred_csv("DATE,USREC\n", 10)


# --- load_series ----------------------------------------------------------------


class TestLoadSeries:
    def test_loads_fixture_with_missing_values(self, fixtures_dir):
        series = load_series("T10Y3M", data_dir=fixtures_dir)

        assert series.name == "T10Y3M"
        assert len(series) == 5
        assert series.isna().sum() == 2  # one "." and one empty cell
        assert series.loc["2020-01-07"] == -0.25

    def test_accepts_string_path(self, fixtures_dir):
        series = load_series("USREC", data_dir=str(fixtures_dir))
        assert series.tolist() == [0.0, 1.0, 1.0]

    def test_uses_env_var_data_dir_by_default(self, monkeypatch, fixtures_dir):
        monkeypatch.setenv(DATA_DIR_ENV_VAR, str(fixtures_dir))
        assert len(load_series("USREC")) == 3

    def test_missing_file_raises_with_hint(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="download_data.py"):
            load_series("DGS10", data_dir=tmp_path)

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_series("DGS10", data_dir=tmp_path / "does-not-exist")

    def test_empty_file_raises(self, tmp_path):
        (tmp_path / "DGS10.csv").write_text("")
        with pytest.raises(ValueError, match="is empty"):
            load_series("DGS10", data_dir=tmp_path)

    def test_malformed_file_raises(self, tmp_path):
        (tmp_path / "DGS10.csv").write_text("<html>Service unavailable</html>\n")
        with pytest.raises(ValueError, match="unexpected columns"):
            load_series("DGS10", data_dir=tmp_path)

    def test_unknown_series_id_raises_before_reading(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown series id"):
            load_series("../secrets", data_dir=tmp_path)

    @pytest.mark.parametrize("bad_id", [None, 42, ["T10Y3M"]])
    def test_non_string_series_id_raises(self, bad_id, fixtures_dir):
        with pytest.raises(TypeError):
            load_series(bad_id, data_dir=fixtures_dir)


# --- load_series_frame ----------------------------------------------------------


class TestLoadSeriesFrame:
    def test_outer_joins_mixed_frequencies(self, fixtures_dir):
        frame = load_series_frame(["T10Y3M", "USREC"], data_dir=fixtures_dir)

        assert list(frame.columns) == ["T10Y3M", "USREC"]
        assert frame.index.name == "date"
        assert frame.index.is_monotonic_increasing
        # 5 daily dates + 3 month starts, no overlap.
        assert len(frame) == 8
        assert frame.loc["2020-02-01", "USREC"] == 1.0
        assert pd.isna(frame.loc["2020-02-01", "T10Y3M"])
        assert frame.loc["2020-01-02", "T10Y3M"] == 1.5
        assert pd.isna(frame.loc["2020-01-02", "USREC"])

    def test_keeps_requested_column_order(self, fixtures_dir):
        frame = load_series_frame(("USREC", "T10Y3M"), data_dir=fixtures_dir)
        assert list(frame.columns) == ["USREC", "T10Y3M"]

    def test_single_series(self, fixtures_dir):
        frame = load_series_frame(["USREC"], data_dir=fixtures_dir)
        assert isinstance(frame, pd.DataFrame)
        assert frame.shape == (3, 1)

    def test_overlapping_dates_are_aligned(self, tmp_path):
        (tmp_path / "DGS2.csv").write_text("DATE,DGS2\n2020-01-01,1\n2020-01-02,2\n")
        (tmp_path / "DGS10.csv").write_text(
            "DATE,DGS10\n2020-01-02,20\n2020-01-03,30\n"
        )

        frame = load_series_frame(["DGS2", "DGS10"], data_dir=tmp_path)

        assert len(frame) == 3
        assert frame.loc["2020-01-02"].tolist() == [2.0, 20.0]

    def test_empty_list_raises(self, fixtures_dir):
        with pytest.raises(ValueError, match="at least one"):
            load_series_frame([], data_dir=fixtures_dir)

    def test_duplicates_raise(self, fixtures_dir):
        with pytest.raises(ValueError, match="duplicates"):
            load_series_frame(["USREC", "T10Y3M", "USREC"], data_dir=fixtures_dir)

    def test_single_string_raises(self, fixtures_dir):
        with pytest.raises(TypeError, match="not a single string"):
            load_series_frame("USREC", data_dir=fixtures_dir)

    def test_missing_file_raises(self, fixtures_dir):
        with pytest.raises(FileNotFoundError, match="DGS30"):
            load_series_frame(["USREC", "DGS30"], data_dir=fixtures_dir)


# --- load_download_date ---------------------------------------------------------


class TestLoadDownloadDate:
    def write_metadata(self, directory: Path, content: str) -> Path:
        (directory / "metadata.json").write_text(content, encoding="utf-8")
        return directory

    def test_reads_date_from_timestamp(self, tmp_path):
        self.write_metadata(tmp_path, '{"downloaded_at": "2026-09-16T21:49:06+00:00"}')
        assert load_download_date(tmp_path) == date(2026, 9, 16)

    def test_accepts_plain_date_and_str_path(self, tmp_path):
        self.write_metadata(tmp_path, '{"downloaded_at": "2024-01-31"}')
        assert load_download_date(str(tmp_path)) == date(2024, 1, 31)

    def test_uses_default_data_dir(self, monkeypatch, tmp_path):
        self.write_metadata(tmp_path, '{"downloaded_at": "2025-05-05T00:00:00"}')
        monkeypatch.setenv(DATA_DIR_ENV_VAR, str(tmp_path))
        assert load_download_date() == date(2025, 5, 5)

    def test_committed_metadata_is_readable(self):
        assert load_download_date() >= date(2026, 1, 1)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="download_data.py"):
            load_download_date(tmp_path)

    def test_invalid_json_raises(self, tmp_path):
        self.write_metadata(tmp_path, "{not json")
        with pytest.raises(ValueError, match="not valid JSON"):
            load_download_date(tmp_path)

    def test_non_object_json_raises(self, tmp_path):
        self.write_metadata(tmp_path, '["2026-09-16"]')
        with pytest.raises(ValueError, match="JSON object"):
            load_download_date(tmp_path)

    @pytest.mark.parametrize("content", ["{}", '{"downloaded_at": 20260916}'])
    def test_missing_timestamp_raises(self, tmp_path, content):
        self.write_metadata(tmp_path, content)
        with pytest.raises(ValueError, match="no 'downloaded_at'"):
            load_download_date(tmp_path)

    def test_invalid_timestamp_raises(self, tmp_path):
        self.write_metadata(tmp_path, '{"downloaded_at": "yesterday"}')
        with pytest.raises(ValueError, match="Invalid 'downloaded_at'"):
            load_download_date(tmp_path)


# --- filter_by_date -------------------------------------------------------------


class TestFilterByDate:
    def test_bounds_are_inclusive(self, daily_series):
        result = filter_by_date(daily_series, "2020-01-03", "2020-01-05")
        assert result.index.strftime("%Y-%m-%d").tolist() == [
            "2020-01-03",
            "2020-01-04",
            "2020-01-05",
        ]

    def test_no_bounds_returns_everything(self, daily_series):
        pd.testing.assert_series_equal(filter_by_date(daily_series), daily_series)

    def test_only_start(self, daily_series):
        result = filter_by_date(daily_series, start="2020-01-09")
        assert result.tolist() == [8.0, 9.0]

    def test_only_end(self, daily_series):
        result = filter_by_date(daily_series, end="2020-01-02")
        assert result.tolist() == [0.0, 1.0]

    def test_start_equals_end(self, daily_series):
        result = filter_by_date(daily_series, "2020-01-04", "2020-01-04")
        assert result.tolist() == [3.0]

    @pytest.mark.parametrize(
        "bound",
        [
            "2020-01-04",
            date(2020, 1, 4),
            datetime(2020, 1, 4),
            pd.Timestamp("2020-01-04"),
        ],
    )
    def test_accepts_date_like_bounds(self, daily_series, bound):
        assert filter_by_date(daily_series, bound, bound).tolist() == [3.0]

    def test_works_on_dataframe(self, daily_series):
        frame = daily_series.to_frame().assign(other=lambda df: df["value"] * 2)
        result = filter_by_date(frame, "2020-01-02", "2020-01-03")
        assert isinstance(result, pd.DataFrame)
        assert result["other"].tolist() == [2.0, 4.0]

    def test_works_on_unsorted_index(self, daily_series):
        shuffled = daily_series.iloc[[5, 0, 9, 2]]
        result = filter_by_date(shuffled, "2020-01-01", "2020-01-06")
        assert sorted(result.tolist()) == [0.0, 2.0, 5.0]

    def test_range_between_observations_is_empty(self):
        monthly = pd.Series(
            [1.0, 2.0], index=pd.DatetimeIndex(["2020-01-01", "2020-02-01"])
        )
        assert filter_by_date(monthly, "2020-01-10", "2020-01-20").empty

    @pytest.mark.parametrize(
        ("start", "end"),
        [("1990-01-01", "1990-12-31"), ("2030-01-01", None), (None, "1900-01-01")],
    )
    def test_range_outside_data_is_empty(self, daily_series, start, end):
        result = filter_by_date(daily_series, start, end)
        assert result.empty
        assert isinstance(result, pd.Series)
        assert result.name == daily_series.name

    def test_empty_input_returns_empty(self):
        empty = pd.Series([], index=pd.DatetimeIndex([]), dtype="float64")
        assert filter_by_date(empty, "2020-01-01", "2020-12-31").empty

    def test_does_not_modify_input(self, daily_series):
        original = daily_series.copy()
        result = filter_by_date(daily_series, "2020-01-01", "2020-01-02")
        result.iloc[0] = 999.0
        pd.testing.assert_series_equal(daily_series, original)

    def test_start_after_end_raises(self, daily_series):
        with pytest.raises(ValueError, match="is after end"):
            filter_by_date(daily_series, "2020-01-05", "2020-01-01")

    @pytest.mark.parametrize("bad", ["not-a-date", "2020-02-30", ""])
    def test_unparseable_bound_raises(self, daily_series, bad):
        with pytest.raises(ValueError, match="not a valid date"):
            filter_by_date(daily_series, start=bad)
        with pytest.raises(ValueError, match="not a valid date"):
            filter_by_date(daily_series, end=bad)

    @pytest.mark.parametrize("bad", [20200101, 2020.0, ["2020-01-01"]])
    def test_unsupported_bound_type_raises(self, daily_series, bad):
        with pytest.raises(TypeError, match="start must be"):
            filter_by_date(daily_series, start=bad)

    @pytest.mark.parametrize("bad", [[1, 2, 3], {"a": 1}, None])
    def test_non_pandas_input_raises(self, bad):
        with pytest.raises(TypeError, match="Series or DataFrame"):
            filter_by_date(bad, "2020-01-01")

    def test_non_datetime_index_raises(self):
        with pytest.raises(TypeError, match="DatetimeIndex"):
            filter_by_date(pd.Series([1, 2, 3]), "2020-01-01")


# --- committed snapshots --------------------------------------------------------


def test_committed_snapshots_load():
    frame = load_series_frame(SERIES_IDS, data_dir=REPO_ROOT / "data" / "raw")

    assert list(frame.columns) == list(SERIES_IDS)
    assert frame.index.min() < pd.Timestamp("1990-01-01")
    assert (frame.notna().sum() > 1000).all()
    assert set(frame["USREC"].dropna().unique()) <= {0.0, 1.0}

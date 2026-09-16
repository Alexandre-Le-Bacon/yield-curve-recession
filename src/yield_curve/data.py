"""Load, merge and filter FRED time series from local CSV snapshots.

All functions here are pure: they read local files only and never call the network.
The snapshots are produced by ``scripts/download_data.py``.
"""

import io
import json
import os
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path
from typing import TypeVar

import pandas as pd

SERIES_IDS: tuple[str, ...] = (
    "T10Y3M",  # 10-year minus 3-month Treasury spread, daily
    "USREC",  # NBER recession indicator (1 = recession), monthly
    "DGS3MO",  # 3-month Treasury yield, daily
    "DGS2",  # 2-year Treasury yield, daily
    "DGS5",  # 5-year Treasury yield, daily
    "DGS10",  # 10-year Treasury yield, daily
    "DGS30",  # 30-year Treasury yield, daily
)

DATA_DIR_ENV_VAR = "YIELD_CURVE_DATA_DIR"
METADATA_FILENAME = "metadata.json"

# FRED has used both names for the date column over time.
_DATE_COLUMNS = ("observation_date", "DATE")
# FRED writes missing observations as "." (older files) or as an empty cell.
_MISSING_MARKERS = ("", ".")

DateLike = str | date | datetime | pd.Timestamp
SeriesOrFrame = TypeVar("SeriesOrFrame", pd.Series, pd.DataFrame)


def get_default_data_dir() -> Path:
    """Return the directory holding the CSV snapshots.

    The ``YIELD_CURVE_DATA_DIR`` environment variable wins when it is set and
    non-empty. Otherwise the function walks up from this file to the project root
    (the first directory containing ``pyproject.toml``) and returns its ``data/raw``
    folder. If no project root is found (e.g. a non-editable install), it falls back
    to ``data/raw`` under the current working directory.

    Returns:
        Absolute path of the data directory. It is not guaranteed to exist.
    """
    env_value = os.environ.get(DATA_DIR_ENV_VAR, "").strip()
    if env_value:
        return Path(env_value).expanduser().resolve()

    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent / "data" / "raw"
    return (Path.cwd() / "data" / "raw").resolve()


DEFAULT_DATA_DIR: Path = get_default_data_dir()


def _validate_series_id(series_id: object) -> str:
    """Check that ``series_id`` is one of the known FRED series ids."""
    if not isinstance(series_id, str):
        raise TypeError(f"series_id must be a string, got {type(series_id).__name__}.")
    if series_id not in SERIES_IDS:
        known = ", ".join(SERIES_IDS)
        raise ValueError(f"Unknown series id {series_id!r}. Expected one of: {known}.")
    return series_id


def parse_fred_csv(csv_text: str, series_id: str) -> pd.Series:
    """Parse the content of a FRED ``fredgraph.csv`` file into a clean series.

    Args:
        csv_text: Raw CSV text with a date column (``observation_date`` or ``DATE``)
            and one value column named after the series.
        series_id: FRED series id expected as the value column name.

    Returns:
        Float series named ``series_id``, indexed by a sorted ``DatetimeIndex``
        named ``date``. Missing observations (``.`` or empty cells) become ``NaN``.
        A file with a header but no rows gives an empty series.

    Raises:
        TypeError: If ``series_id`` is not a string.
        ValueError: If ``series_id`` is unknown, the text is empty, the columns are
            not the expected ones, or a date or value cannot be parsed, or a date
            appears twice.
    """
    series_id = _validate_series_id(series_id)
    if not csv_text.strip():
        raise ValueError(f"CSV for {series_id} is empty.")

    try:
        raw = pd.read_csv(io.StringIO(csv_text), dtype=str, keep_default_na=False)
    except pd.errors.ParserError as error:
        raise ValueError(f"CSV for {series_id} is malformed.") from error
    columns = list(raw.columns)
    if len(columns) != 2 or columns[0] not in _DATE_COLUMNS or columns[1] != series_id:
        raise ValueError(
            f"CSV for {series_id} has unexpected columns {columns}; expected "
            f"a date column ({' or '.join(_DATE_COLUMNS)}) followed by {series_id!r}."
        )
    date_column = columns[0]

    try:
        dates = pd.to_datetime(raw[date_column].str.strip(), format="%Y-%m-%d")
    except ValueError as error:
        raise ValueError(f"CSV for {series_id} contains an invalid date.") from error
    if dates.isna().any():
        raise ValueError(f"CSV for {series_id} contains a missing date.")

    values_text = raw[series_id].str.strip()
    values_text = values_text.where(~values_text.isin(_MISSING_MARKERS), None)
    try:
        values = pd.to_numeric(values_text).astype("float64")
    except ValueError as error:
        raise ValueError(
            f"CSV for {series_id} contains a non-numeric value."
        ) from error

    series = pd.Series(
        values.to_numpy(),
        index=pd.DatetimeIndex(dates, name="date"),
        name=series_id,
        dtype="float64",
    )
    if series.index.has_duplicates:
        raise ValueError(f"CSV for {series_id} contains duplicate dates.")
    return series.sort_index()


def load_series(series_id: str, data_dir: str | Path | None = None) -> pd.Series:
    """Load one FRED series from its local CSV snapshot.

    Args:
        series_id: FRED series id, one of ``SERIES_IDS`` (e.g. ``"T10Y3M"``).
        data_dir: Directory containing ``<series_id>.csv``. Defaults to
            ``get_default_data_dir()``, evaluated at call time.

    Returns:
        Float series indexed by date, see ``parse_fred_csv``.

    Raises:
        TypeError: If ``series_id`` is not a string.
        ValueError: If ``series_id`` is unknown or the file is malformed.
        FileNotFoundError: If the snapshot file does not exist.
    """
    series_id = _validate_series_id(series_id)
    directory = Path(data_dir) if data_dir is not None else get_default_data_dir()
    path = directory / f"{series_id}.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"No snapshot for {series_id} at {path}. "
            "Run `uv run python scripts/download_data.py` to download the data."
        )
    return parse_fred_csv(path.read_text(encoding="utf-8"), series_id)


def load_series_frame(
    series_ids: Sequence[str], data_dir: str | Path | None = None
) -> pd.DataFrame:
    """Load several FRED series and align them on a common date index.

    Series with different frequencies (daily and monthly) are outer-joined: every
    date present in at least one series is kept, and gaps are ``NaN``.

    Args:
        series_ids: FRED series ids to load, in the desired column order.
        data_dir: Directory containing the CSV snapshots. Defaults to
            ``get_default_data_dir()``.

    Returns:
        DataFrame with one float column per series, indexed by a sorted
        ``DatetimeIndex`` named ``date``.

    Raises:
        TypeError: If ``series_ids`` is a single string instead of a sequence.
        ValueError: If ``series_ids`` is empty or contains duplicates, or if any
            series fails to load (see ``load_series``).
        FileNotFoundError: If a snapshot file does not exist.
    """
    if isinstance(series_ids, str):
        raise TypeError(
            "series_ids must be a sequence of ids, not a single string: "
            f"use [{series_ids!r}]."
        )
    ids = list(series_ids)
    if not ids:
        raise ValueError("series_ids must contain at least one series id.")
    duplicates = sorted({sid for sid in ids if ids.count(sid) > 1})
    if duplicates:
        raise ValueError(f"series_ids contains duplicates: {duplicates}.")

    frame = pd.concat([load_series(sid, data_dir) for sid in ids], axis=1, sort=True)
    frame.index.name = "date"
    return frame


def load_download_date(data_dir: str | Path | None = None) -> date:
    """Read the date the snapshots were downloaded from ``metadata.json``.

    Args:
        data_dir: Directory containing ``metadata.json``, as written by
            ``scripts/download_data.py``. Defaults to ``get_default_data_dir()``.

    Returns:
        Calendar date of the ``downloaded_at`` timestamp, as recorded (UTC for
        files written by the download script).

    Raises:
        FileNotFoundError: If ``metadata.json`` does not exist.
        ValueError: If the file is not valid JSON, is not a JSON object, or has no
            valid ISO 8601 ``downloaded_at`` timestamp.
    """
    directory = Path(data_dir) if data_dir is not None else get_default_data_dir()
    path = directory / METADATA_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"No metadata file at {path}. "
            "Run `uv run python scripts/download_data.py` to download the data."
        )
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not valid JSON.") from error
    if not isinstance(metadata, dict):
        raise ValueError(f"{path} must contain a JSON object.")

    downloaded_at = metadata.get("downloaded_at")
    if not isinstance(downloaded_at, str):
        raise ValueError(f"{path} has no 'downloaded_at' timestamp.")
    try:
        return datetime.fromisoformat(downloaded_at).date()
    except ValueError as error:
        raise ValueError(
            f"Invalid 'downloaded_at' timestamp in {path}: {downloaded_at!r}."
        ) from error


def _to_timestamp(value: DateLike | None, name: str) -> pd.Timestamp | None:
    """Convert an optional date-like bound to a ``Timestamp``."""
    if value is None:
        return None
    if not isinstance(value, str | date):
        raise TypeError(
            f"{name} must be a date string, date, datetime or Timestamp, "
            f"got {type(value).__name__}."
        )
    try:
        timestamp = pd.Timestamp(value)
    except ValueError as error:
        raise ValueError(f"{name} is not a valid date: {value!r}.") from error
    if pd.isna(timestamp):
        raise ValueError(f"{name} is not a valid date: {value!r}.")
    return timestamp


def filter_by_date(
    data: SeriesOrFrame,
    start: DateLike | None = None,
    end: DateLike | None = None,
) -> SeriesOrFrame:
    """Keep the rows whose date falls within ``[start, end]`` (both inclusive).

    Args:
        data: Series or DataFrame indexed by a ``DatetimeIndex``.
        start: First date to keep, or ``None`` for no lower bound.
        end: Last date to keep, or ``None`` for no upper bound.

    Returns:
        A new object of the same type with only the rows in range. If no row falls
        in the range (including a range entirely outside the data), the result is
        empty rather than an error. The input is never modified.

    Raises:
        TypeError: If ``data`` is not a Series/DataFrame, its index is not a
            ``DatetimeIndex``, or a bound has an unsupported type.
        ValueError: If a bound cannot be parsed as a date, or ``start`` is after
            ``end``.
    """
    if not isinstance(data, pd.Series | pd.DataFrame):
        raise TypeError(
            f"data must be a pandas Series or DataFrame, got {type(data).__name__}."
        )
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError(
            f"data must be indexed by a DatetimeIndex, got {type(data.index).__name__}."
        )

    start_ts = _to_timestamp(start, "start")
    end_ts = _to_timestamp(end, "end")
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        raise ValueError(f"start ({start_ts.date()}) is after end ({end_ts.date()}).")

    lower = start_ts if start_ts is not None else pd.Timestamp.min
    upper = end_ts if end_ts is not None else pd.Timestamp.max
    return data.loc[(data.index >= lower) & (data.index <= upper)].copy()

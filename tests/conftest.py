from pathlib import Path

import pandas as pd
import pytest

from yield_curve.data import DATA_DIR_ENV_VAR

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    """Directory with small FRED-style CSVs (T10Y3M daily, USREC monthly)."""
    return FIXTURES_DIR


@pytest.fixture(autouse=True)
def _no_data_dir_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate every test from a YIELD_CURVE_DATA_DIR set in the developer's shell."""
    monkeypatch.delenv(DATA_DIR_ENV_VAR, raising=False)


@pytest.fixture
def daily_series() -> pd.Series:
    """Ten consecutive days of data starting on 2020-01-01."""
    index = pd.date_range("2020-01-01", periods=10, freq="D", name="date")
    return pd.Series(range(10), index=index, name="value", dtype="float64")

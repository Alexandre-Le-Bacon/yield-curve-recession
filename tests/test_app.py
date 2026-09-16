"""Smoke tests: every page of the Streamlit app runs without raising.

The app is always started from ``app/Home.py``, as ``streamlit run app/Home.py``
does, so the ``from loaders import ...`` lines are tested in the same conditions as
in production. The pages read the committed snapshots in ``data/raw/`` (no network).
"""

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_DIR = Path(__file__).resolve().parents[1] / "app"
HOME = str(APP_DIR / "Home.py")
TIMEOUT_SECONDS = 30


@pytest.fixture(autouse=True)
def _fresh_loaders_import():
    """Force each test to import ``loaders`` again, through Streamlit's sys.path."""
    assert str(APP_DIR) not in sys.path
    sys.modules.pop("loaders", None)
    yield
    sys.modules.pop("loaders", None)


def run_home() -> AppTest:
    app = AppTest.from_file(HOME, default_timeout=TIMEOUT_SECONDS)
    app.run()
    return app


def test_loaders_is_not_importable_without_streamlit():
    with pytest.raises(ModuleNotFoundError):
        __import__("loaders")


def test_home_page_runs():
    app = run_home()

    assert not app.exception
    assert app.title[0].value == "Can the bond market predict recessions?"
    assert [metric.label for metric in app.metric] == [
        "Data range",
        "Yield curve inversions",
        "Recessions",
    ]
    assert app.sidebar.caption[0].value.startswith("Data: FRED, downloaded ")


def test_understand_the_curve_page_runs():
    app = run_home()
    app.switch_page("pages/1_Understand_the_curve.py").run()

    assert not app.exception
    assert app.select_slider(key="curve_month").value is not None
    assert len(app.button) == 4


def test_quick_pick_button_moves_the_month_slider():
    app = run_home()
    app.switch_page("pages/1_Understand_the_curve.py").run()

    app.button(key="pick_2006_12").click().run()

    assert not app.exception
    assert str(app.select_slider(key="curve_month").value) == "2006-12-01"
    assert app.metric[0].value == "Inverted"


def test_the_signal_page_runs():
    app = run_home()
    app.switch_page("pages/2_The_signal.py").run()

    assert not app.exception
    assert [metric.label for metric in app.metric] == [
        "Inversion episodes in range",
        "Recession periods in range",
    ]


def test_min_months_slider_changes_the_episode_count():
    app = run_home()
    app.switch_page("pages/2_The_signal.py").run()
    default_count = int(app.metric[0].value)

    app.slider[1].set_value(1).run()

    assert not app.exception
    assert int(app.metric[0].value) > default_count

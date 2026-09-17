"""Smoke tests: every page of the Streamlit app runs without raising.

The app is always started from ``app/Home.py``, as ``streamlit run app/Home.py``
does, so the ``from loaders import ...`` lines are tested in the same conditions as
in production. The pages read the committed snapshots in ``data/raw/`` (no network).
"""

import json
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


def test_home_page_conclusion_is_built_from_the_evaluation():
    from yield_curve.data import load_series
    from yield_curve.model import build_dataset, evaluate

    app = run_home()

    evaluation = evaluate(build_dataset(load_series("T10Y3M"), load_series("USREC")))
    subheaders = [subheader.value for subheader in app.subheader]
    assert "So, can it?" in subheaders
    conclusion = next(m.value for m in app.markdown if "Brier score" in m.value)
    assert f"{evaluation.model_brier:.3f}" in conclusion
    assert f"{evaluation.baseline_brier:.3f}" in conclusion
    assert conclusion.startswith(
        ("**Yes, so far.**", "**Partly.**", "**Not reliably.**")
    )


def test_home_page_links_to_every_page():
    app = run_home()

    # Streamlit links pages by URL name: "pages/2_The_signal.py" -> "The_signal".
    linked = {link.proto.page for link in app.get("page_link")}
    pages = {path.stem.split("_", 1)[1] for path in (APP_DIR / "pages").glob("*.py")}
    assert pages
    assert pages <= linked


def test_understand_the_curve_page_runs():
    app = run_home()
    app.switch_page("pages/1_Understand_the_curve.py").run()

    assert not app.exception
    assert app.select_slider(key="curve_month").value is not None
    assert len(app.button) == 4
    captions = [caption.value for caption in app.caption]
    assert "A recession started in January 2008, 13 months later." in captions
    assert "No recession in the 24 months that followed." in captions


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


def test_signal_chart_uses_a_date_axis():
    # Regression: legend-only traces made Plotly pick a numeric axis (empty chart).
    app = run_home()
    app.switch_page("pages/2_The_signal.py").run()

    (chart,) = app.get("plotly_chart")
    layout = json.loads(chart.proto.spec)["layout"]
    assert layout["xaxis"]["type"] == "date"


def test_model_page_runs():
    app = run_home()
    app.switch_page("pages/3_The_model.py").run()

    assert not app.exception
    readings, count = app.metric[:-1], app.metric[-1]
    assert readings[0].label.startswith("Recession within 12 months, based on ")
    # A second reading only appears when the data ends in the middle of a month.
    for partial in readings[1:]:
        assert partial.label.startswith("So far in ")
        assert partial.label.endswith("(partial data)")
    for reading in readings:
        assert 0 <= int(reading.value.rstrip("%")) <= 100
    assert count.label == "Recession months in the test period"
    assert len(app.table) == 1


def test_model_chart_marks_the_out_of_sample_period():
    app = run_home()
    app.switch_page("pages/3_The_model.py").run()

    (chart,) = app.get("plotly_chart")
    layout = json.loads(chart.proto.spec)["layout"]
    assert layout["xaxis"]["type"] == "date"
    annotations = [annotation["text"] for annotation in layout["annotations"]]
    assert "Right of this line: years the model never saw" in annotations
    assert layout["yaxis"]["range"] == [0, 1]


def test_data_and_method_page_runs():
    app = run_home()
    app.switch_page("pages/4_Data_and_method.py").run()

    assert not app.exception
    text = " ".join(element.value for element in app.markdown)
    assert "https://fred.stlouisfed.org/" in text
    assert "https://www.nber.org/" in text
    assert "embargo" in text
    assert "Not financial advice" in app.warning[0].value

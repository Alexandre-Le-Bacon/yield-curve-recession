"""Cached data access shared by every page of the app.

Streamlit adds the folder of the main script (``app/``) to ``sys.path``, so pages
can ``from loaders import ...`` when the app is started with
``streamlit run app/Home.py``, locally or in Docker.
"""

from datetime import date

import pandas as pd
import streamlit as st

from yield_curve.data import load_download_date, load_series, load_series_frame
from yield_curve.features import YIELD_CURVE_SERIES, monthly_spread
from yield_curve.model import (
    CurrentReading,
    Evaluation,
    build_dataset,
    current_reading,
    evaluate,
    fit_evaluation_model,
    predict_probability,
)

# Inversions shorter than this are treated as noise in the key facts and charts.
DEFAULT_MIN_MONTHS = 3


@st.cache_data
def get_monthly_spread() -> pd.Series:
    """Monthly average of the 10-year minus 3-month spread, in percentage points."""
    return monthly_spread(load_series("T10Y3M"))


@st.cache_data
def get_recession_indicator() -> pd.Series:
    """Monthly NBER recession indicator (1 = recession month)."""
    return load_series("USREC")


@st.cache_data
def get_yields() -> pd.DataFrame:
    """Daily Treasury yields for the maturities of the yield curve, in percent."""
    return load_series_frame(list(YIELD_CURVE_SERIES))


@st.cache_data
def get_evaluation() -> Evaluation:
    """Out-of-sample evaluation of the model with the default cutoff and horizon."""
    return evaluate(build_dataset(load_series("T10Y3M"), get_recession_indicator()))


@st.cache_data
def get_evaluation_probabilities() -> pd.Series:
    """Monthly probabilities from the model trained before the cutoff only."""
    dataset = build_dataset(load_series("T10Y3M"), get_recession_indicator())
    model = fit_evaluation_model(dataset)
    return predict_probability(model, get_monthly_spread().dropna())


@st.cache_data
def get_current_reading() -> CurrentReading:
    """Latest reading from the model refitted on all labelled months."""
    return current_reading(load_series("T10Y3M"), get_recession_indicator())


@st.cache_data
def get_download_date() -> date:
    """Date the data snapshots were downloaded from FRED."""
    return load_download_date()


def show_data_source() -> None:
    """Show where the data comes from and when it was downloaded, in the sidebar."""
    st.sidebar.caption(f"Data: FRED, downloaded {get_download_date():%Y-%m-%d}")

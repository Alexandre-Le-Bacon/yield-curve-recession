import streamlit as st
from loaders import (
    DEFAULT_MIN_MONTHS,
    get_monthly_spread,
    get_recession_indicator,
    show_data_source,
)

from yield_curve.features import inversion_episodes, recession_periods

st.set_page_config(page_title="Can the bond market predict recessions?", page_icon="📉")
show_data_source()

spread = get_monthly_spread().dropna()
first_month, last_month = spread.index[0], spread.index[-1]
episodes = inversion_episodes(spread, min_months=DEFAULT_MIN_MONTHS)
# Only count recessions that started after the spread data begins, so each one
# can be compared with the curve before it.
recessions = [
    period
    for period in recession_periods(get_recession_indicator())
    if period[0] > first_month
]

st.title("Can the bond market predict recessions?")

st.markdown(
    """
When the US government borrows money, it pays an interest rate called a **yield**,
and lenders normally ask for a higher yield to lend for 10 years than for 3 months.
The gap between those two rates is the **spread**; when it turns negative, the yield
curve is said to be **inverted**, a sign that investors expect the economy to weaken.
This app checks, with 40+ years of official data, whether inversions really came
before US **recessions** (periods when the economy shrinks, as dated by economists
at the NBER).
"""
)

st.subheader("Key facts")
col1, col2, col3 = st.columns(3)
col1.metric("Data range", f"{first_month:%Y} – {last_month:%Y}")
col2.metric("Yield curve inversions", len(episodes))
col3.metric("Recessions", len(recessions))
col3.caption(
    f"Recessions that started after {first_month:%B %Y}, when the spread data "
    "begins. The signal page also shades a recession already under way then."
)
st.caption(
    f"An inversion counts here only if the monthly spread stayed below zero for at "
    f"least {DEFAULT_MIN_MONTHS} months in a row, to ignore short blips."
)

st.subheader("Explore")
st.page_link(
    "pages/1_Understand_the_curve.py",
    label="Understand the curve: what a normal and an inverted yield curve look like",
    icon="📈",
)
st.page_link(
    "pages/2_The_signal.py",
    label="The signal: 40 years of the spread, next to every recession",
    icon="🔔",
)
st.page_link(
    "pages/3_The_model.py",
    label="The model: today's 12-month recession probability, and how well it works",
    icon="🎯",
)
st.page_link(
    "pages/4_Data_and_method.py",
    label="Data and method: sources, how the model is built, and its limits",
    icon="📚",
)

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from loaders import INVERSION_COLOR, SERIES_COLOR, get_yields, show_data_source

from yield_curve.features import MATURITY_YEARS, yield_curve_on

st.set_page_config(page_title="Understand the curve", page_icon="📈")
show_data_source()

MONTH_KEY = "curve_month"
# Notable months, with what happened next.
NOTABLE_MONTHS: dict[str, tuple[date, str]] = {
    "Dot-com bubble": (
        date(2000, 6, 1),
        "A recession started in March 2001, after the dot-com crash.",
    ),
    "Before the financial crisis": (
        date(2006, 12, 1),
        "A recession started in January 2008 and lasted until June 2009.",
    ),
    "Before Covid": (
        date(2019, 8, 1),
        "A short recession hit in March–April 2020, caused by the pandemic.",
    ),
    "Record inversion": (
        date(2023, 6, 1),
        "No recession had been declared by the end of the data: a false alarm so far.",
    ),
}

st.title("Understand the yield curve")
st.markdown(
    """
The **yield curve** plots the interest rate (yield) the US government pays against
how long it borrows for, from 3 months to 30 years.

- **Normal curve (upward slope):** lenders ask for more to lock their money away
  for longer, so long-term yields are higher than short-term ones.
- **Inverted curve (downward slope):** short-term yields are *higher* than long-term
  ones. Investors expect interest rates, and often the economy, to fall.
"""
)

yields = get_yields()
# Start when both ends of the 10Y-3M spread are available.
first_day = yields[["DGS3MO", "DGS10"]].dropna().index[0]
months = [
    timestamp.date()
    for timestamp in pd.date_range(
        first_day.to_period("M").to_timestamp(), yields.index[-1], freq="MS"
    )
]
if MONTH_KEY not in st.session_state:
    st.session_state[MONTH_KEY] = months[-1]


def pick_month(month: date) -> None:
    st.session_state[MONTH_KEY] = month


st.subheader("Jump to a notable moment")
for column, (label, (month, what_next)) in zip(
    st.columns(len(NOTABLE_MONTHS)), NOTABLE_MONTHS.items(), strict=True
):
    column.button(
        f"{label} ({month:%b %Y})",
        key=f"pick_{month:%Y_%m}",
        on_click=pick_month,
        args=(month,),
        width="stretch",
    )
    column.caption(what_next)

selected = st.select_slider(
    "Or pick any month",
    options=months,
    key=MONTH_KEY,
    format_func=lambda month: f"{month:%b %Y}",
)

# Use the last available day of the selected month.
as_of, curve = yield_curve_on(yields, pd.Timestamp(selected) + pd.offsets.MonthEnd(0))
spread = curve["10Y"] - curve["3M"]
inverted = spread < 0

col1, col2 = st.columns(2)
col1.metric("Curve shape", "Inverted" if inverted else "Normal")
col2.metric("10-year minus 3-month spread", f"{spread:+.2f} pp")
st.caption(
    f"Yields on {as_of:%d %B %Y}, the last trading day with data in that month. "
    "pp = percentage points."
)

x_years = [MATURITY_YEARS[label] for label in curve.index]
figure = go.Figure(
    go.Scatter(
        x=x_years,
        y=curve.to_numpy(),
        mode="lines+markers+text",
        line={"color": INVERSION_COLOR if inverted else SERIES_COLOR, "width": 2},
        marker={"size": 9},
        text=[f"{value:.2f}%" for value in curve],
        textposition="top center",
        customdata=list(curve.index),
        hovertemplate="%{customdata}: %{y:.2f}%<extra></extra>",
    )
)
figure.update_layout(
    title=f"US Treasury yield curve, {as_of:%B %Y}",
    xaxis={
        "title": "Time to maturity (log scale)",
        "type": "log",
        "tickvals": x_years,
        "ticktext": list(curve.index),
    },
    yaxis={"title": "Yield (%)", "rangemode": "tozero"},
    margin={"t": 60},
    showlegend=False,
)
st.plotly_chart(figure)

if len(curve) < len(MATURITY_YEARS):
    st.caption(
        "Some maturities have no data for this month (for example, the 30-year bond "
        "was not issued between 2002 and 2006)."
    )

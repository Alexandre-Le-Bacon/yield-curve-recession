import plotly.graph_objects as go
import streamlit as st
from loaders import (
    DEFAULT_MIN_MONTHS,
    INVERSION_COLOR,
    RECESSION_COLOR,
    SERIES_COLOR,
    get_monthly_spread,
    get_recession_indicator,
    show_data_source,
)

from yield_curve.data import filter_by_date
from yield_curve.features import inversion_episodes, recession_periods

st.set_page_config(page_title="The signal", page_icon="🔔")
show_data_source()

st.title("The signal: spread vs. recessions")
st.markdown(
    """
The line shows the **spread**: the 10-year yield minus the 3-month yield, averaged
each month. Below the dashed zero line, the curve is **inverted**.
Grey bands are **recessions**; orange bands are inversion episodes.
Look at what comes *after* each orange band.
"""
)

spread = get_monthly_spread().dropna()
usrec = get_recession_indicator()
first_day, last_day = spread.index[0].date(), spread.index[-1].date()

start, end = st.slider(
    "Date range",
    min_value=first_day,
    max_value=last_day,
    value=(first_day, last_day),
    format="MMM YYYY",
)
min_months = st.slider(
    "Minimum length of an inversion (months)",
    min_value=1,
    max_value=6,
    value=DEFAULT_MIN_MONTHS,
    help="Shorter inversions are ignored.",
)
st.caption(
    "A one-month dip below zero can be noise, so by default an inversion only counts "
    f"if it lasts at least {DEFAULT_MIN_MONTHS} months: move the slider to see how "
    "many short episodes appear or disappear."
)

spread_in_range = filter_by_date(spread, start, end)
if spread_in_range.empty:
    st.warning("No data in this date range. Widen the range to see the chart.")
    st.stop()

episodes = inversion_episodes(spread_in_range, min_months=min_months)
recessions = recession_periods(filter_by_date(usrec, start, end))

col1, col2 = st.columns(2)
col1.metric("Inversion episodes in range", len(episodes))
col2.metric("Recession periods in range", len(recessions))

figure = go.Figure()
for periods, color in ((recessions, RECESSION_COLOR), (episodes, INVERSION_COLOR)):
    for period_start, period_end in periods:
        figure.add_vrect(
            x0=period_start,
            x1=period_end,
            fillcolor=color,
            opacity=0.25,
            line_width=0,
            layer="below",
        )
# Invisible traces so the shaded bands appear in the legend.
for name, color in (("Recession", RECESSION_COLOR), ("Inversion", INVERSION_COLOR)):
    figure.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker={"size": 12, "symbol": "square", "color": color, "opacity": 0.5},
            name=name,
        )
    )
figure.add_trace(
    go.Scatter(
        x=spread_in_range.index,
        y=spread_in_range.to_numpy(),
        mode="lines",
        line={"color": SERIES_COLOR, "width": 2},
        name="10Y − 3M spread",
        hovertemplate="%{x|%b %Y}: %{y:+.2f} pp<extra></extra>",
    )
)
figure.add_hline(y=0, line_dash="dash", line_width=1, line_color=RECESSION_COLOR)
figure.update_layout(
    title="10-year minus 3-month Treasury spread (monthly average)",
    xaxis_title=None,
    yaxis_title="Spread (percentage points)",
    hovermode="x unified",
    legend={"orientation": "h", "y": -0.12},
    margin={"t": 60},
)
st.plotly_chart(figure)

st.caption(
    "Sources: FRED series T10Y3M (spread) and USREC (recessions dated by the NBER). "
    "Recessions are shaded by month."
)

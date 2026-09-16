"""Shared Plotly chart style for the app pages (presentation only, no data logic)."""

from collections.abc import Iterable

import pandas as pd
import plotly.graph_objects as go

# Chart colors (reference data-viz palette): series, inversion and recession marks.
SERIES_COLOR = "#2a78d6"
INVERSION_COLOR = "#eb6834"
RECESSION_COLOR = "#8a8984"

Period = tuple[pd.Timestamp, pd.Timestamp]


def time_series_figure(title: str, yaxis_title: str) -> go.Figure:
    """Empty figure with the layout shared by the monthly time-series charts."""
    figure = go.Figure()
    figure.update_layout(
        title=title,
        # Explicit: legend-only traces have no dates for Plotly to infer the axis from.
        xaxis={"type": "date", "title": None},
        yaxis_title=yaxis_title,
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.12},
        margin={"t": 60},
    )
    return figure


def shade_periods(
    figure: go.Figure, periods: Iterable[Period], color: str, name: str
) -> None:
    """Shade each period as a vertical band and add one legend entry for them."""
    for start, end in periods:
        figure.add_vrect(
            x0=start, x1=end, fillcolor=color, opacity=0.25, line_width=0, layer="below"
        )
    # Invisible trace so the shaded bands appear in the legend.
    figure.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker={"size": 12, "symbol": "square", "color": color, "opacity": 0.5},
            name=name,
            hoverinfo="skip",
        )
    )

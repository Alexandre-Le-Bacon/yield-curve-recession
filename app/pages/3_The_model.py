import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from charts import RECESSION_COLOR, SERIES_COLOR, shade_periods, time_series_figure
from loaders import (
    get_current_reading,
    get_evaluation,
    get_evaluation_probabilities,
    get_recession_indicator,
    show_data_source,
)

from yield_curve.data import filter_by_date
from yield_curve.features import recession_periods
from yield_curve.model import DEFAULT_HORIZON

st.set_page_config(page_title="The model", page_icon="🎯")
show_data_source()

reading = get_current_reading()
evaluation = get_evaluation()
usrec = get_recession_indicator()

st.title("The model: a recession probability")
st.markdown(
    f"""
The model looks at a single number, the **spread** (10-year minus 3-month yield)
averaged over a month. From past data, it learns how often a recession started
within the next {DEFAULT_HORIZON} months at each spread level, and turns any spread
into a **probability** between 0% and 100%: the lower the spread, the higher the
probability. Technically, it is a *logistic regression*, one of the simplest
classification models.
"""
)

st.subheader("Today's reading")
latest = reading.latest_complete
col1, col2 = st.columns(2)
col1.metric(
    f"Recession within {DEFAULT_HORIZON} months, based on {latest.month:%B %Y}",
    f"{latest.probability:.0%}",
)
col1.caption(f"Average spread in {latest.month:%B %Y}: {latest.spread:+.2f} pp.")
if reading.in_progress is not None:
    partial = reading.in_progress
    col2.metric(
        f"So far in {partial.month:%B %Y} (partial data)",
        f"{partial.probability:.0%}",
    )
    col2.caption(
        f"Month in progress: spread averaged over the days up to "
        f"{reading.data_until:%d %B %Y} ({partial.spread:+.2f} pp). "
        "It may still change."
    )
st.caption(
    f"This reading comes from a model trained on every month with a known outcome "
    f"({reading.trained_from:%b %Y} – {reading.trained_until:%b %Y}). The chart below "
    f"uses a model trained only up to {evaluation.train_end:%b %Y}: "
    "the chart tests the method, the headline gives today's reading."
)

st.subheader("Would it have worked? Testing on years the model never saw")
probabilities = get_evaluation_probabilities()
last_month = probabilities.index[-1]

figure = time_series_figure(
    f"Probability of a recession within {DEFAULT_HORIZON} months "
    f"(model trained up to {evaluation.train_end:%b %Y})",
    "Probability",
)
figure.add_vrect(
    x0=evaluation.test_start,
    x1=last_month + pd.offsets.MonthEnd(0),
    fillcolor=SERIES_COLOR,
    opacity=0.07,
    line_width=0,
    layer="below",
)
shade_periods(
    figure,
    recession_periods(usrec.loc[probabilities.index[0] :]),
    RECESSION_COLOR,
    "Recession",
)
figure.add_trace(
    go.Scatter(
        x=probabilities.index,
        y=probabilities.to_numpy(),
        mode="lines",
        line={"color": SERIES_COLOR, "width": 2},
        name="Model probability",
        hovertemplate="%{x|%b %Y}: %{y:.0%}<extra></extra>",
    )
)
figure.add_vline(
    x=evaluation.test_start, line_dash="dash", line_width=1, line_color=SERIES_COLOR
)
figure.add_annotation(
    x=evaluation.test_start,
    y=1,
    yref="paper",
    text="Right of this line: years the model never saw",
    showarrow=False,
    xanchor="left",
    yanchor="bottom",
    xshift=6,
)
figure.update_yaxes(range=[0, 1], tickformat=".0%")
st.plotly_chart(figure)
st.caption(
    f"Left of the line, the model was trained on {evaluation.train_start:%b %Y} – "
    f"{evaluation.train_end:%b %Y}. The {DEFAULT_HORIZON} months before "
    f"{evaluation.test_start:%B %Y} "
    "are left out of training (an *embargo*), because their outcome was not yet known "
    "at that time. See the Data and method page."
)

st.subheader("How good is it?")
st.markdown(
    f"""
We compare the model with a **naive baseline** that ignores the spread and always
predicts {evaluation.base_rate:.0%}, the share of training months that were followed
by a recession. Both are scored on the test period ({evaluation.test_start:%b %Y} –
{evaluation.test_end:%b %Y}), the months right of the line.

- **ROC AUC** measures how well the scores rank months: 0.5 is a coin flip, 1.0 means
  every month before a recession got a higher probability than every other month.
- **Brier score** is the average squared gap between the predicted probability and
  what happened (1 or 0): lower is better, 0 is perfect.
"""
)

test_usrec = filter_by_date(usrec, evaluation.test_start, evaluation.test_end)
test_recessions = recession_periods(test_usrec)
table_column, count_column = st.columns([3, 2])
table_column.table(
    pd.DataFrame(
        {
            "Model": [f"{evaluation.model_auc:.2f}", f"{evaluation.model_brier:.3f}"],
            "Naive baseline": [
                f"{evaluation.baseline_auc:.2f}",
                f"{evaluation.baseline_brier:.3f}",
            ],
        },
        index=pd.Index(["ROC AUC (higher is better)", "Brier score (lower is better)"]),
    )
)
count_column.metric("Recession months in the test period", int(test_usrec.sum()))
count_column.caption(
    f"These scores rest on only {len(test_recessions)} recessions, so they are "
    "fragile: one more or one less would change them a lot."
)

ranks_better = evaluation.model_auc > evaluation.baseline_auc
more_accurate = evaluation.model_brier < evaluation.baseline_brier
st.markdown(
    f"""
**In plain words:** on years it never saw, the model ranks months
{"better" if ranks_better else "no better"} than chance,
{"and" if ranks_better == more_accurate else "but"} its probabilities are
{"more" if more_accurate else "less"} accurate than simply always predicting
{evaluation.base_rate:.0%}. The spread is a warning signal worth watching, not a
precise forecast: see the limitations on the Data and method page.
"""
)

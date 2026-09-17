import streamlit as st
from loaders import (
    DEFAULT_MIN_MONTHS,
    get_evaluation,
    get_monthly_spread,
    get_recession_indicator,
    show_data_source,
)

from yield_curve.features import Period, signal_track_record

st.set_page_config(page_title="Can the bond market predict recessions?", page_icon="📉")
show_data_source()

FOLLOW_UP_MONTHS = 24

spread = get_monthly_spread().dropna()
first_month, last_month = spread.index[0], spread.index[-1]
# Recessions are counted only if they started after the spread data begins, so each
# one can be compared with the curve before it.
record = signal_track_record(
    spread,
    get_recession_indicator(),
    within_months=FOLLOW_UP_MONTHS,
    min_months=DEFAULT_MIN_MONTHS,
)
evaluation = get_evaluation()


def describe_periods(periods: tuple[Period, ...]) -> str:
    return ", ".join(f"{start:%b %Y} – {end:%b %Y}" for start, end in periods)


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
col2.metric("Yield curve inversions", len(record.episodes))
col3.metric("Recessions", len(record.recessions))
col3.caption(
    f"Recessions that started after {first_month:%B %Y}, when the spread data "
    "begins. The signal page also shades a recession already under way then."
)
st.caption(
    f"An inversion counts here only if the monthly spread stayed below zero for at "
    f"least {DEFAULT_MIN_MONTHS} months in a row, to ignore short blips."
)

st.subheader("So, can it?")

all_preceded = len(record.preceded) == len(record.recessions)
model_more_accurate = evaluation.model_brier < evaluation.baseline_brier
if all_preceded and model_more_accurate and not record.false_alarms:
    verdict = "Yes, so far."
elif all_preceded or model_more_accurate:
    verdict = "Partly."
else:
    verdict = "Not reliably."

signal_sentence = (
    (
        f"All {len(record.recessions)}"
        if all_preceded
        else f"{len(record.preceded)} of the {len(record.recessions)}"
    )
    + f" recessions since {first_month:%Y} came less than "
    f"{FOLLOW_UP_MONTHS} months after the yield curve inverted."
)
if record.false_alarms:
    count = len(record.false_alarms)
    alarm_sentence = (
        f"But {count} inversion{'s' if count > 1 else ''} "
        f"({describe_periods(record.false_alarms)}) "
        f"{'were' if count > 1 else 'was'} not followed by a recession within "
        f"{FOLLOW_UP_MONTHS} months: a false alarm."
    )
else:
    alarm_sentence = (
        f"No inversion has been a false alarm within {FOLLOW_UP_MONTHS} months."
    )
if record.pending:
    pending_sentence = (
        f"The most recent inversion ({describe_periods(record.pending[-1:])}) is too "
        "recent to judge."
    )
else:
    pending_sentence = ""
model_sentence = (
    f"On years it never saw ({evaluation.test_start:%Y}–{evaluation.test_end:%Y}), "
    f"the model's recession probabilities were "
    f"{'more' if model_more_accurate else 'less'} accurate than a naive guess that "
    f"always says {evaluation.base_rate:.0%} (Brier score "
    f"{evaluation.model_brier:.3f} vs {evaluation.baseline_brier:.3f}, lower is "
    "better)."
)
st.markdown(
    f"**{verdict}** {signal_sentence} {alarm_sentence} {pending_sentence}\n\n"
    f"{model_sentence} The inverted yield curve is a warning sign worth watching, "
    "not a reliable forecast on its own."
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

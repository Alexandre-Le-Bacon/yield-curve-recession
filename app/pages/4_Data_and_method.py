import streamlit as st
from loaders import (
    DEFAULT_MIN_MONTHS,
    get_current_reading,
    get_download_date,
    get_evaluation,
    get_monthly_spread,
    get_recession_indicator,
    show_data_source,
)

from yield_curve.features import inversion_episodes, recession_after, recession_periods
from yield_curve.model import DEFAULT_HORIZON

st.set_page_config(page_title="Data and method", page_icon="📚")
show_data_source()

FOLLOW_UP_MONTHS = 24

spread = get_monthly_spread().dropna()
usrec = get_recession_indicator()
evaluation = get_evaluation()
reading = get_current_reading()
last_usrec_month = usrec.dropna().index[-1]
recessions = [p for p in recession_periods(usrec) if p[0] > spread.index[0]]
train_recessions = [p for p in recessions if p[0] <= evaluation.train_end]

st.title("Data and method")

st.header("Data sources")
FRED_SERIES = {
    "T10Y3M": "10-year minus 3-month Treasury yield spread, daily",
    "USREC": "Recession indicator: 1 for a recession month, 0 otherwise",
    "DGS3MO": "3-month Treasury yield, daily (yield curve page)",
    "DGS2": "2-year Treasury yield, daily (yield curve page)",
    "DGS5": "5-year Treasury yield, daily (yield curve page)",
    "DGS10": "10-year Treasury yield, daily (yield curve page)",
    "DGS30": "30-year Treasury yield, daily (yield curve page)",
}
series_rows = "\n".join(
    f"| [{series_id}](https://fred.stlouisfed.org/series/{series_id}) | {description} |"
    for series_id, description in FRED_SERIES.items()
)
st.markdown(
    f"""
All data comes from **[FRED](https://fred.stlouisfed.org/)**, the free public
database of the Federal Reserve Bank of St. Louis. The files were downloaded on
**{get_download_date():%d %B %Y}** and are stored in the repository, so the app never
needs the internet and always shows the same numbers.

| Series | What it is |
| --- | --- |
{series_rows}

Recession dates are set by the
**[NBER](https://www.nber.org/research/business-cycle-dating)** (National Bureau of
Economic Research), a private non-profit whose Business Cycle Dating Committee is the
reference for US recession dates.
"""
)

st.header("Method")
st.markdown(
    f"""
**Feature.** For each month, the average of the daily spread over that month.

**Target.** For each month *t*, the answer to: "does at least one recession month
occur in the next {DEFAULT_HORIZON} months, from *t*+1 to *t*+{DEFAULT_HORIZON}?"
(1 = yes, 0 = no). The current month is not included: the question is whether a
recession is *coming*.

**Why the last {DEFAULT_HORIZON} months have no target.** For a recent month, part of
the next {DEFAULT_HORIZON} months has not happened yet, so the answer is unknown.
Filling it with 0 would quietly assume "no recession", so these months are left out
of training and testing (this is called avoiding *look-ahead*). With recession data
up to {last_usrec_month:%B %Y}, the last month with a known target is
{reading.trained_until:%B %Y}.

**Model.** A logistic regression with one input (the spread), from scikit-learn,
with a fixed random seed, so the results are identical on every run.

**Train/test split in time, with an embargo.** The model is trained on the past and
tested on the future, never shuffled:

- training: {evaluation.train_start:%b %Y} – {evaluation.train_end:%b %Y}
  ({evaluation.n_train} months);
- **embargo**: the {DEFAULT_HORIZON} months before the test period are used for
  neither. Their target depends on recessions after the cutoff, which nobody could
  have known at the time;
- test: {evaluation.test_start:%b %Y} – {evaluation.test_end:%b %Y}
  ({evaluation.n_test} months).

**Evaluation.** On the test period, the model is compared with a naive baseline that
always predicts the training share of positive months ({evaluation.base_rate:.0%}),
using ROC AUC (ranking quality) and the Brier score (probability accuracy).

**Today's reading.** Separately, the model is refitted on every month with a known
target and applied to the last complete month. The month in progress is shown apart,
because its average only covers part of the month.
"""
)

st.header("Limitations")

longest = max(
    inversion_episodes(spread, min_months=DEFAULT_MIN_MONTHS),
    key=lambda episode: episode[1] - episode[0],
)
after = recession_after(usrec, longest[0], months=FOLLOW_UP_MONTHS)
if after.first_recession_month is not None:
    longest_outcome = (
        f"was followed by a recession starting in {after.first_recession_month:%B %Y}"
    )
elif after.complete:
    longest_outcome = (
        f"was **not** followed by a recession within {FOLLOW_UP_MONTHS} months of "
        "its start, a false alarm for the signal"
    )
else:
    longest_outcome = (
        f"has not been followed by a recession so far (recession data ends in "
        f"{last_usrec_month:%B %Y})"
    )

st.markdown(
    f"""
- **Very few recessions.** Only {len(recessions)} recessions started since
  {spread.index[0]:%B %Y}: {len(train_recessions)} in the training period and
  the rest in the test period. Any score computed on so few events is fragile.
- **The longest inversion in the data** ({longest[0]:%b %Y} – {longest[1]:%b %Y})
  {longest_outcome}. The signal has worked before, but it is not a law.
- **Recessions are dated late.** The NBER announces the start of a recession months
  after it has begun (for example, the December 2007 peak that started the 2008–2009
  recession was only announced in December 2008). The most recent zeros in the
  recession data could still turn into ones, which would change the latest targets.
- **One input only.** The model ignores everything except the spread: inflation,
  jobs, central bank policy, global shocks such as a pandemic.
- **The world changes.** Relationships learned from past decades may not hold in
  the future.
"""
)

st.warning(
    "**Not financial advice.** This app is an educational project. It does not "
    "recommend buying or selling anything, and it should not be used to make "
    "investment or business decisions.",
    icon="⚠️",
)

st.header("Reproduce the results")
st.markdown(
    """
Everything runs locally with [uv](https://docs.astral.sh/uv/), which installs the
exact package versions recorded in `uv.lock`:
"""
)
st.code(
    """git clone <repository-url>
cd yield-curve-recession
uv sync                                  # exact dependencies from uv.lock
uv run pytest --cov=src                  # run the test suite
uv run streamlit run app/Home.py         # start the app on http://localhost:8501""",
    language="bash",
)
st.markdown(
    """
With the committed data files, the numbers match this page exactly. To use the latest
data from FRED instead (the results will change), run
`uv run python scripts/download_data.py` before starting the app.
"""
)

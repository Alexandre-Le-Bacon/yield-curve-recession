# Can the bond market predict recessions?

A Streamlit app that explores one of the best-known warning signals in economics:
the **inverted yield curve**.

- A **yield** is the annual interest rate the US government pays to borrow money
  by issuing Treasury bonds.
- Lenders usually ask for a higher yield to lend for longer, so the 10-year yield is
  normally above the 3-month yield. The difference between the two is the
  **spread**.
- When the spread turns negative, short-term borrowing costs more than long-term
  borrowing. This is called a **yield curve inversion**, and it has come before
  every US recession of the past several decades.

The app shows the history of the spread against official recession periods, and fits
a simple model estimating the probability of a recession within the next 12 months.

> Final project for the HEC course *Tooling for the Data Scientist*.

## Status

| Part                                    | Status          |
| --------------------------------------- | --------------- |
| Data download, loading and filtering    | ✅ done          |
| Tests and continuous integration        | ✅ done          |
| Features: spread, inversions, target    | ✅ done          |
| Exploration pages (app v0)              | ✅ done          |
| Recession probability model and method  | ✅ done          |
| Docker image                            | 🚧 coming next  |

## Quick start

Requirements: [uv](https://docs.astral.sh/uv/). uv installs the right Python
version (3.11) and all dependencies for you. You don't need to install Python yourself.

```bash
# 1. Install uv (macOS / Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone the repository and install the exact dependency versions from uv.lock
git clone <repository-url>
cd yield-curve-recession
uv sync

# 3. Run the app, then open http://localhost:8501
uv run streamlit run app/Home.py
```

Always start the app from the repository root with `app/Home.py` as the entry point:
Streamlit then adds `app/` to the import path, which the pages need to import the
shared `app/loaders.py`.

## The app

| Page                    | What it shows |
| ----------------------- | ------------- |
| **Home**                | The question, a short explanation for non-finance readers, and key facts: data range, number of yield curve inversions (lasting at least 3 months) and number of recessions since 1982. |
| **Understand the curve**| The shape of the yield curve (3-month to 30-year yields) for any month since 1982, labelled *normal* or *inverted*. Buttons jump to notable moments (dot-com bubble, before the financial crisis, before Covid, record inversion), each with what happened in the next 24 months, computed from the recession data. |
| **The signal**          | The 10-year minus 3-month spread over time, with a zero line, recessions shaded in grey and inversion episodes in orange. Sliders filter the date range and set the minimum length of an inversion (1 to 6 months). |
| **The model**           | Today's 12-month recession probability (last complete month, plus the month in progress shown separately), a chart of the probability over time from a model that never saw the years after 2006, and its evaluation against a naive baseline. |
| **Data and method**     | Data sources with links, download date, target definition, train/test split with embargo, limitations, a *not financial advice* note, and how to reproduce the results. |

The sidebar shows when the data was downloaded from FRED. Data loading is cached
with `st.cache_data`, so the CSV files are read once per server process.

## Model and method

The question the model answers: *given the spread this month, what is the probability
that a recession month occurs within the next 12 months?*

- **Feature:** the monthly average of the daily 10-year minus 3-month spread.
- **Target:** 1 if at least one month in `(t, t+12]` is a recession month (USREC),
  else 0. The last 12 months of recession data have no known target and are dropped,
  never filled with 0 (no look-ahead).
- **Model:** scikit-learn `LogisticRegression` with a fixed `random_state`.
- **Split:** by time, never shuffled. The cutoff month is 2006-12: training uses
  months up to 2005-12, testing starts in 2007-01. The 12 months in between are an
  **embargo**: their targets depend on recessions after the cutoff, so using them for
  training would leak future information. The embargo length follows the horizon.
- **Evaluation:** ROC AUC and Brier score on the test period, against a naive
  baseline that always predicts the training base rate.
- **Current reading:** a separate model refitted on all labelled months, applied to
  the last complete month; the month in progress is reported apart.

Results with the committed snapshot (downloaded 2026-09-16), test period
Jan 2007 – Aug 2025:

| Metric                          | Model | Naive baseline |
| ------------------------------- | ----- | -------------- |
| ROC AUC (higher is better)      | 0.59  | 0.50           |
| Brier score (lower is better)   | 0.190 | 0.153          |

The model ranks months better than chance, but its probabilities are less accurate
than the baseline. These scores rest on only two recessions (2008–2009 and 2020), and
the long 2022–2024 inversion was not followed by a recession. The app states this
openly: the spread is a warning signal, not a precise forecast. This is not financial
advice.

## Development

Run the same checks as the CI:

```bash
uv run ruff check .                  # lint
uv run ruff format --check .         # formatting (use `uv run ruff format .` to fix)
uv run pytest --cov=src              # tests + coverage (fails below 90%)
```

The test suite has two parts:

- `tests/test_data.py`, `tests/test_features.py`, `tests/test_model.py`: unit tests
  of the pure functions in `src/`, on small synthetic data (plus one check that the
  committed snapshots load and the default model runs).
- `tests/test_app.py`: smoke tests that run every page with Streamlit's `AppTest`,
  starting from `app/Home.py` like `streamlit run` does, on the committed snapshots.

Every push to `main` and every pull request runs these checks on GitHub Actions
(`.github/workflows/ci.yml`).

## Data

All data comes from [FRED](https://fred.stlouisfed.org/), the public database of the
Federal Reserve Bank of St. Louis:

| Series   | Description                                        | Frequency |
| -------- | -------------------------------------------------- | --------- |
| `T10Y3M` | 10-year minus 3-month Treasury spread              | daily     |
| `USREC`  | Recession indicator (1 = recession, dated by NBER) | monthly   |
| `DGS3MO` | 3-month Treasury yield                             | daily     |
| `DGS2`   | 2-year Treasury yield                              | daily     |
| `DGS5`   | 5-year Treasury yield                              | daily     |
| `DGS10`  | 10-year Treasury yield                             | daily     |
| `DGS30`  | 30-year Treasury yield                             | daily     |

Snapshots are committed in `data/raw/`, so the app works offline and results are
reproducible. The download date is recorded in `data/raw/metadata.json`. To refresh
the data:

```bash
uv run python scripts/download_data.py
```

The app reads data from `data/raw/` by default. Set the `YIELD_CURVE_DATA_DIR`
environment variable to use another directory.

## Project layout

```
app/                     Streamlit UI, no business logic
  Home.py                  entry point: question, explanation, key facts
  loaders.py               cached data loading and model results shared by the pages
  charts.py                shared Plotly chart style (date axis, shaded periods)
  pages/                   one file per page
src/yield_curve/         Pure Python logic, no Streamlit imports
  data.py                  load, merge and filter FRED series; download date
  features.py              monthly spread, inversions, recession periods and target
  model.py                 dataset, time split with embargo, logistic regression,
                           evaluation and current reading
scripts/download_data.py Refresh the CSV snapshots from FRED
data/raw/                Committed CSV snapshots + metadata.json
tests/                   pytest suite (no network access)
```

## License

MIT, see [LICENSE](LICENSE).

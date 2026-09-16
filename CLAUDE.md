# CLAUDE.md — yield-curve-recession

## Project goal
Streamlit app answering one question: **"Can the bond market predict recessions?"**
It explores the US yield curve (10-year minus 3-month Treasury spread), shows that
curve inversions preceded past US recessions, and fits a simple model estimating the
probability of a recession within the next 12 months.

This is the graded final project of the HEC course "Tooling for the Data Scientist".
Graders are software engineers, **not finance experts**. Engineering quality matters
as much as the app itself: tests, CI, reproducibility, documentation, clean Git history.

## Audience and tone
- App text is in English, plain language, no unexplained jargon.
- Every finance concept (yield, spread, inversion) gets a one-sentence explanation
  the first time it appears.

## Stack
- Python 3.11, dependencies managed with **uv** (`pyproject.toml` + committed `uv.lock`).
- Streamlit (multipage app), pandas, plotly, scikit-learn.
- Dev tools: pytest, pytest-cov, ruff (lint + format).
- CI: GitHub Actions. Packaging: Docker.

## Repository layout
```
app/
  Home.py                  # Streamlit entry point (UI only)
  pages/                   # one file per page
src/yield_curve/
  data.py                  # loading, cleaning, filtering (pure functions)
  features.py              # spread, inversion detection, target construction
  model.py                 # model training and prediction
scripts/
  download_data.py         # refreshes the CSV snapshots from FRED
data/raw/                  # committed CSV snapshots (the app never calls the network)
tests/                     # pytest, mirrors src/ structure
.github/workflows/ci.yml
Dockerfile, .dockerignore, pyproject.toml, uv.lock, README.md
```

## Data
- Source: FRED (Federal Reserve Bank of St. Louis), public CSV endpoint, no API key:
  `https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES_ID>`
- Series: `T10Y3M` (10Y-3M spread, daily), `USREC` (NBER recession indicator, monthly),
  `DGS3MO`, `DGS2`, `DGS5`, `DGS10`, `DGS30` (Treasury yields by maturity, daily).
- Missing values may appear as empty cells or `.`: handle both.
- Snapshots are committed in `data/raw/`; the download script records the download date.
- The app reads local files only.

## Engineering rules
- **Separation of concerns:** no Streamlit import in `src/`. The UI calls pure functions.
- Type hints on every function; Google-style docstrings on every public function.
- **Tests are mandatory** for every function in `src/`, especially data loading and
  filtering (explicit course requirement): normal cases, edge cases (empty range,
  missing values, dates outside the data), and invalid inputs (`pytest.raises`).
- Tests use small synthetic DataFrames or fixtures in `tests/fixtures/`, never the network.
- Run `uv run ruff check . && uv run ruff format --check . && uv run pytest --cov=src`
  before declaring a task done. Everything must pass.
- Model: time-based train/test split only (no random shuffling, no look-ahead leakage).
  Fixed random seeds.
- Keep it simple: no premature abstractions, no extra dependencies without asking.

## Git workflow
- One feature = one branch (`feat/...`, `fix/...`, `docs/...`, `ci/...`).
- Small, atomic commits with Conventional Commit messages
  (`feat: add date range filter`, `test: cover missing values in loader`).
- Never commit to `main` directly. **Never push**: the user reviews and pushes.
- Never commit secrets, virtual environments or cache files.

## Docker
- Base image `python:3.11-slim`, install with uv from the lockfile, run as non-root user.
- Expose port 8501, add a HEALTHCHECK, start with
  `streamlit run app/Home.py --server.port=8501 --server.address=0.0.0.0`.
- The image must work on both `linux/amd64` and `linux/arm64`.

## Working with the user
- The user is learning these tools: briefly explain what you did and why after each task.
- Propose a plan before large changes and wait for approval.

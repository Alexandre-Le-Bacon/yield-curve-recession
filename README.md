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
| Exploration pages                       | 🚧 coming next  |
| Recession probability model             | 🚧 coming next  |
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

## Development

Run the same checks as the CI:

```bash
uv run ruff check .                  # lint
uv run ruff format --check .         # formatting (use `uv run ruff format .` to fix)
uv run pytest --cov=src              # tests + coverage (fails below 90%)
```

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
app/                     Streamlit UI (Home.py + pages/)
src/yield_curve/         Pure Python logic, no Streamlit imports
  data.py                  load, merge and filter FRED series
  features.py              spread, inversions, recession target (coming next)
  model.py                 recession probability model (coming next)
scripts/download_data.py Refresh the CSV snapshots from FRED
data/raw/                Committed CSV snapshots + metadata.json
tests/                   pytest suite (no network access)
```

## License

MIT, see [LICENSE](LICENSE).

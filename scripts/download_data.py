"""Download the FRED series used by the app into local CSV snapshots.

Usage:
    uv run python scripts/download_data.py [--output-dir DIR]

Every series is downloaded and validated in memory first. Files are written only if
all downloads succeed, so a network failure never leaves a half-updated snapshot.
The download date is recorded in ``metadata.json`` next to the CSV files.
"""

import argparse
import json
import os
import sys
import urllib.request
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import URLError

from yield_curve.data import SERIES_IDS, get_default_data_dir, parse_fred_csv

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
# FRED may reject the default "Python-urllib" agent, so identify the project instead.
USER_AGENT = "yield-curve-recession/0.1"
TIMEOUT_SECONDS = 60

Fetcher = Callable[[str], str]


def build_url(series_id: str) -> str:
    """Return the public FRED CSV URL for a series.

    Args:
        series_id: FRED series id, e.g. ``"T10Y3M"``.

    Returns:
        The download URL.
    """
    return FRED_CSV_URL.format(series_id=series_id)


def fetch_csv(series_id: str) -> str:
    """Download the raw CSV text of one FRED series.

    Args:
        series_id: FRED series id.

    Returns:
        The response body decoded as UTF-8.

    Raises:
        URLError: On network errors or HTTP error statuses.
        TimeoutError: If the server does not answer in time.
    """
    request = urllib.request.Request(
        build_url(series_id), headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return response.read().decode("utf-8")


def download_all(
    series_ids: Sequence[str], fetch: Fetcher = fetch_csv
) -> dict[str, str]:
    """Download and validate several series, keeping the results in memory.

    Args:
        series_ids: FRED series ids to download.
        fetch: Function returning the CSV text for a series id. Injected in tests
            so that they never touch the network.

    Returns:
        Mapping from series id to its validated CSV text.

    Raises:
        ValueError: If a downloaded file is malformed or has no observations.
        URLError: On network errors.
        TimeoutError: If a request times out.
    """
    downloads: dict[str, str] = {}
    for series_id in series_ids:
        print(f"Downloading {series_id}...", flush=True)
        text = fetch(series_id)
        series = parse_fred_csv(text, series_id)
        if series.dropna().empty:
            raise ValueError(f"Downloaded {series_id} has no observations.")
        downloads[series_id] = text
    return downloads


def _write_atomically(path: Path, content: str) -> None:
    """Write ``content`` to a temporary file, then rename it over ``path``."""
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, path)


def write_snapshots(
    downloads: dict[str, str], output_dir: Path, downloaded_at: datetime
) -> None:
    """Write the CSV snapshots and their ``metadata.json`` to ``output_dir``.

    CSV files are stored exactly as returned by FRED; cleaning happens at load time.

    Args:
        downloads: Mapping from series id to validated CSV text.
        output_dir: Destination directory, created if needed.
        downloaded_at: Download timestamp recorded in the metadata.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    series_info: dict[str, dict[str, object]] = {}
    for series_id, text in downloads.items():
        _write_atomically(output_dir / f"{series_id}.csv", text)
        series = parse_fred_csv(text, series_id).dropna()
        series_info[series_id] = {
            "url": build_url(series_id),
            "observations": len(series),
            "first_date": series.index.min().date().isoformat(),
            "last_date": series.index.max().date().isoformat(),
        }
    metadata = {
        "source": "FRED, Federal Reserve Bank of St. Louis",
        "downloaded_at": downloaded_at.isoformat(timespec="seconds"),
        "series": series_info,
    }
    _write_atomically(
        output_dir / "metadata.json", json.dumps(metadata, indent=2) + "\n"
    )


def main(argv: Sequence[str] | None = None, fetch: Fetcher = fetch_csv) -> int:
    """Run the download script.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).
        fetch: Function returning the CSV text for a series id.

    Returns:
        Process exit code: 0 on success, 1 if any download failed.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=get_default_data_dir(),
        help="directory for the CSV snapshots (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    try:
        downloads = download_all(SERIES_IDS, fetch=fetch)
    except (URLError, TimeoutError, ValueError) as error:
        print(f"Download failed, no file was written: {error}", file=sys.stderr)
        return 1

    write_snapshots(downloads, args.output_dir, downloaded_at=datetime.now(UTC))
    print(f"Wrote {len(downloads)} series to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

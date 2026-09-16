import json
from datetime import UTC, datetime
from urllib.error import URLError

import pytest

import download_data
from yield_curve.data import SERIES_IDS, load_series_frame


def fake_csv(series_id: str) -> str:
    return (
        f"observation_date,{series_id}\n2020-01-01,.\n2020-01-02,1.5\n2020-01-03,1.6\n"
    )


class FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def test_build_url():
    assert download_data.build_url("T10Y3M") == (
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10Y3M"
    )


def test_fetch_csv_sends_user_agent_and_timeout(monkeypatch):
    calls = {}

    def fake_urlopen(request, timeout):
        calls["url"] = request.full_url
        calls["user_agent"] = request.get_header("User-agent")
        calls["timeout"] = timeout
        return FakeResponse(b"observation_date,USREC\n2020-01-01,0\n")

    monkeypatch.setattr(download_data.urllib.request, "urlopen", fake_urlopen)

    text = download_data.fetch_csv("USREC")

    assert text.startswith("observation_date,USREC")
    assert calls["url"].endswith("?id=USREC")
    assert calls["user_agent"] == "yield-curve-recession/0.1"
    assert calls["timeout"] == download_data.TIMEOUT_SECONDS


def test_download_all_returns_validated_text():
    downloads = download_data.download_all(["DGS2", "DGS10"], fetch=fake_csv)
    assert list(downloads) == ["DGS2", "DGS10"]
    assert downloads["DGS2"] == fake_csv("DGS2")


def test_download_all_rejects_malformed_response():
    with pytest.raises(ValueError, match="unexpected columns"):
        download_data.download_all(["DGS2"], fetch=lambda _: "<html>error</html>")


def test_download_all_rejects_series_without_observations():
    with pytest.raises(ValueError, match="no observations"):
        download_data.download_all(
            ["DGS2"], fetch=lambda sid: f"DATE,{sid}\n2020-01-01,.\n"
        )


def test_write_snapshots_writes_csv_and_metadata(tmp_path):
    output_dir = tmp_path / "raw"
    downloads = {"USREC": fake_csv("USREC")}
    when = datetime(2026, 9, 16, 12, 0, tzinfo=UTC)

    download_data.write_snapshots(downloads, output_dir, downloaded_at=when)

    assert (output_dir / "USREC.csv").read_text() == fake_csv("USREC")
    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert metadata["downloaded_at"] == "2026-09-16T12:00:00+00:00"
    assert metadata["series"]["USREC"] == {
        "url": download_data.build_url("USREC"),
        "observations": 2,
        "first_date": "2020-01-02",
        "last_date": "2020-01-03",
    }
    assert not list(output_dir.glob(".*.tmp"))


def test_main_writes_loadable_snapshots(tmp_path, capsys):
    exit_code = download_data.main(["--output-dir", str(tmp_path)], fetch=fake_csv)

    assert exit_code == 0
    frame = load_series_frame(SERIES_IDS, data_dir=tmp_path)
    assert list(frame.columns) == list(SERIES_IDS)
    assert f"Wrote {len(SERIES_IDS)} series" in capsys.readouterr().out


@pytest.mark.parametrize(
    "error", [URLError("connection refused"), TimeoutError("timed out")]
)
def test_main_writes_nothing_if_any_download_fails(tmp_path, capsys, error):
    def flaky_fetch(series_id: str) -> str:
        if series_id == "DGS10":
            raise error
        return fake_csv(series_id)

    exit_code = download_data.main(["--output-dir", str(tmp_path)], fetch=flaky_fetch)

    assert exit_code == 1
    assert list(tmp_path.iterdir()) == []
    assert "no file was written" in capsys.readouterr().err

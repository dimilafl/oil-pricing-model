import io
import zipfile
from pathlib import Path

from oil_risk.adapters.gdelt_adapter import GdeltAdapter, GdeltQuery


class DummyResp:
    def __init__(self, text: str = "", content: bytes = b""):
        self.text = text
        self.content = content

    def raise_for_status(self):
        return None


def test_fetch_and_parse(monkeypatch, tmp_path: Path):
    row = "\t".join(
        [
            "1",
            "20240101000000",
            "",
            "",
            "example.com",
            "https://x",
            "Iran tanker news",
            "IRAN;TAX_FNCACT_SANCTION;",
            "",
            "IRAN#",
            "",
            "PERSON1",
            "",
            "ORG1",
            "",
            "-5,0,0,0,0,0,0",
        ]
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("sample.gkg.csv", row + "\n")

    def fake_get(url, timeout=30):
        if url.endswith("lastupdate.txt"):
            return DummyResp(text="1 1 http://data.gdeltproject.org/gdeltv2/sample.gkg.csv.zip")
        return DummyResp(content=buf.getvalue())

    monkeypatch.setattr("oil_risk.adapters.gdelt_adapter.requests.get", fake_get)
    adapter = GdeltAdapter(tmp_path)
    raw, norm = adapter.fetch_and_parse(GdeltQuery(days=9999, max_files=1))
    assert not raw.empty
    assert not norm.empty
    assert "keyword_count" in norm.columns

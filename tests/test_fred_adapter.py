from pathlib import Path

from oil_risk.adapters.fred_adapter import FredAdapter


class DummyResp:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self):
        return None


def test_series_to_dataframe(monkeypatch, tmp_path: Path):
    csv_text = "DATE,DCOILWTICO\n2024-01-01,70\n2024-01-02,71\n"

    def fake_get(*args, **kwargs):
        return DummyResp(csv_text)

    monkeypatch.setattr("oil_risk.adapters.fred_adapter.requests.get", fake_get)
    adapter = FredAdapter(tmp_path)
    df = adapter.series_to_dataframe("DCOILWTICO", force_refresh=True)
    assert {"date", "value", "series_id", "source", "pulled_at"} <= set(df.columns)
    assert len(df) == 2

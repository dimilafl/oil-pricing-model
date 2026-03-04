from datetime import date

from oil_risk.options_flow.polygon import PolygonOptionsFlowProvider


class DummyResp:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "results": [
                {
                    "details": {"contract_type": "put"},
                    "day": {"volume": 120},
                    "implied_volatility": 0.3,
                },
                {
                    "details": {"contract_type": "put"},
                    "day": {"volume": 80},
                    "implied_volatility": None,
                },
                {
                    "details": {"contract_type": "call"},
                    "day": {"volume": 100},
                    "implied_volatility": 0.2,
                },
            ]
        }


def test_polygon_provider_metrics(monkeypatch):
    monkeypatch.setattr("oil_risk.options_flow.polygon.requests.get", lambda *a, **k: DummyResp())
    provider = PolygonOptionsFlowProvider("key")
    out = provider.fetch_daily_metrics("USO", date(2024, 1, 1))
    ratio = out[out["metric_name"] == "put_call_ratio"]["metric_value"].iloc[0]
    iv = out[out["metric_name"] == "implied_vol_proxy"]["metric_value"].iloc[0]
    assert ratio == 2.0
    assert iv == 0.25

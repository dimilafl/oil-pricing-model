import json

from oil_risk.signals_config import load_signals_config


def test_load_signals_config_defaults(tmp_path):
    config = load_signals_config(tmp_path / "missing.json")
    assert config["risk_premium_alert"]["ovx_z_min"] == 1.0
    assert config["tail_risk_alert"]["tail_risk_prob_min"] == 0.5


def test_load_signals_config_overrides(tmp_path):
    path = tmp_path / "signals.json"
    path.write_text(
        json.dumps(
            {
                "risk_premium_alert": {"ovx_z_min": 1.25},
                "tail_risk_alert": {"tail_risk_prob_min": 0.65},
            }
        ),
        encoding="utf-8",
    )
    config = load_signals_config(path)
    assert config["risk_premium_alert"]["ovx_z_min"] == 1.25
    assert config["risk_premium_alert"]["news_risk_min"] == 1.0
    assert config["tail_risk_alert"]["tail_risk_prob_min"] == 0.65

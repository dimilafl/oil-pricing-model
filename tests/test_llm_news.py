from oil_risk.llm.news_classifier import OpenAINewsClassifier


class DummyResp:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_openai_classifier(monkeypatch):
    payload = {
        "output": [
            {
                "content": [
                    {
                        "text": (
                            '{"relevance_score":0.8,"category":"direct_conflict",'
                            '"intensity":2,"entities":{"countries":["Iran"],"orgs":[],"people":[]},'
                            '"short_summary":"Test summary."}'
                        )
                    }
                ]
            }
        ]
    }
    monkeypatch.setattr(
        "oil_risk.llm.news_classifier.requests.post", lambda *a, **k: DummyResp(payload)
    )
    clf = OpenAINewsClassifier("key", model_name="gpt-test")
    out = clf.classify("headline", "{}")
    assert out["category"] == "direct_conflict"
    assert out["model_name"] == "gpt-test"


def test_openai_classifier_parses_json_code_fence(monkeypatch):
    payload = {
        "output": [
            {
                "content": [
                    {
                        "text": '```json\n{"relevance_score":0.2,"category":"other","intensity":0}\n```'
                    }
                ]
            }
        ]
    }
    monkeypatch.setattr(
        "oil_risk.llm.news_classifier.requests.post", lambda *a, **k: DummyResp(payload)
    )
    clf = OpenAINewsClassifier("key")
    out = clf.classify("headline", "{}")
    assert out["category"] == "other"
    assert out["intensity"] == 0

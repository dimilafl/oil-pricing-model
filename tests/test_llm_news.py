from oil_risk.llm.news_classifier import OpenAINewsClassifier


class DummyResp:
    def raise_for_status(self):
        return None

    def json(self):
        return {
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


def test_openai_classifier(monkeypatch):
    monkeypatch.setattr("oil_risk.llm.news_classifier.requests.post", lambda *a, **k: DummyResp())
    clf = OpenAINewsClassifier("key", model_name="gpt-test")
    out = clf.classify("headline", "{}")
    assert out["category"] == "direct_conflict"
    assert out["model_name"] == "gpt-test"

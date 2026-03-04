from __future__ import annotations

import json
from datetime import UTC, datetime

import requests

TAXONOMY = [
    "shipping_disruption",
    "sanctions_enforcement",
    "direct_conflict",
    "diplomacy",
    "energy_infrastructure",
    "other",
]


class OpenAINewsClassifier:
    def __init__(self, api_key: str, model_name: str = "gpt-4.1-mini"):
        self.api_key = api_key
        self.model_name = model_name

    def _extract_json(self, text: str) -> dict:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
        return json.loads(cleaned)

    def classify(self, title: str | None, raw_record_json: str | None) -> dict:
        prompt = {
            "title": title,
            "raw_record": raw_record_json,
            "task": (
                "Classify this energy/geopolitical news item and return strict JSON with keys: "
                "relevance_score (0..1), category, intensity (0..3 integer), "
                "entities (object with countries/orgs/people arrays), short_summary (1-2 sentences)."
            ),
            "allowed_categories": TAXONOMY,
        }
        resp = requests.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model_name,
                "input": json.dumps(prompt),
                "text": {"format": {"type": "json_object"}},
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data.get("output", [{}])[0].get("content", [{}])[0].get("text", "{}")
        parsed = self._extract_json(text)
        parsed["model_name"] = self.model_name
        parsed["created_at"] = datetime.now(UTC)
        return parsed

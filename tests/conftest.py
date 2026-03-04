from __future__ import annotations

import pytest
import requests


@pytest.fixture(autouse=True)
def block_outbound_http(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    if request.node.get_closest_marker("allow_http"):
        return

    def _blocked(*args, **kwargs):
        raise RuntimeError("Outbound HTTP is blocked in tests. Mock requests explicitly.")

    monkeypatch.setattr(requests, "get", _blocked)
    monkeypatch.setattr(requests, "post", _blocked)

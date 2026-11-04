import json

import pytest
from fastapi.testclient import TestClient
from hwsw.repo_paths import FIXTURE_BOARDS_DIR

from hwsw_app.main import create_app
from hwsw_app.settings import AppSettings


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(AppSettings(FIXTURE_BOARDS_DIR, "fixtures")))


def parse_sse(text: str) -> list[tuple[str, dict]]:
    """[(event name, data), ...] from a server-sent-events body."""
    messages = []
    for block in filter(None, text.split("\n\n")):
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        messages.append((lines["event"], json.loads(lines["data"])))
    return messages

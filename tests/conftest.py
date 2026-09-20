from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.db import connect, init_db  # noqa: E402
from app.demo import demo_payload  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("LIDAR_ALIGN_DB", str(db_path))
    conn = connect(str(db_path))
    init_db(conn)
    app.state.db = conn
    yield conn, str(db_path)
    conn.close()


@pytest.fixture()
def client(tmp_db):
    return TestClient(app)


@pytest.fixture()
def demo_pointset(client):
    resp = client.post("/api/pointsets", json=demo_payload())
    assert resp.status_code == 200, resp.text
    return resp.json()["id"]

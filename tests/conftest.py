from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from app import database
import app.main as main


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    monkeypatch.setattr(main, "get_connection", lambda: database.connect(db_path))
    monkeypatch.setenv("SEED_DEMO", "0")
    with TestClient(main.app) as test_client:
        yield test_client

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(os.environ.get("LIDAR_DB_PATH", "data/app.db"))

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    source_name TEXT NOT NULL,
    common_crs TEXT NOT NULL,
    common_length_unit TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    input_fingerprint TEXT NOT NULL,
    created_job_id TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS strips (
    dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    strip_id TEXT NOT NULL,
    label TEXT NOT NULL,
    time_start REAL NOT NULL,
    time_end REAL NOT NULL,
    time_unit TEXT NOT NULL,
    time_epoch TEXT NOT NULL,
    local_crs TEXT NOT NULL,
    local_length_unit TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    PRIMARY KEY (dataset_id, strip_id)
);
CREATE TABLE IF NOT EXISTS points (
    dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    point_id TEXT NOT NULL,
    strip_id TEXT NOT NULL,
    time_value REAL NOT NULL,
    x REAL NOT NULL,
    y REAL NOT NULL,
    z REAL NOT NULL,
    PRIMARY KEY (dataset_id, point_id),
    FOREIGN KEY (dataset_id, strip_id) REFERENCES strips(dataset_id, strip_id)
);
CREATE TABLE IF NOT EXISTS controls (
    dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    control_id TEXT NOT NULL,
    point_id TEXT NOT NULL,
    x REAL NOT NULL,
    y REAL NOT NULL,
    z REAL NOT NULL,
    weight REAL NOT NULL,
    metadata_json TEXT NOT NULL,
    PRIMARY KEY (dataset_id, control_id),
    FOREIGN KEY (dataset_id, point_id) REFERENCES points(dataset_id, point_id)
);
CREATE TABLE IF NOT EXISTS correspondences (
    dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    correspondence_id TEXT NOT NULL,
    left_point_id TEXT NOT NULL,
    right_point_id TEXT NOT NULL,
    weight REAL NOT NULL,
    group_id TEXT,
    metadata_json TEXT NOT NULL,
    PRIMARY KEY (dataset_id, correspondence_id),
    FOREIGN KEY (dataset_id, left_point_id) REFERENCES points(dataset_id, point_id),
    FOREIGN KEY (dataset_id, right_point_id) REFERENCES points(dataset_id, point_id)
);
CREATE TABLE IF NOT EXISTS schemes (
    scheme_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id),
    parent_scheme_id TEXT REFERENCES schemes(scheme_id),
    label TEXT NOT NULL,
    model_version TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('draft','published')),
    config_json TEXT NOT NULL,
    rules_fingerprint TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_job_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    published_at TEXT
);
CREATE TABLE IF NOT EXISTS scheme_controls (
    scheme_id TEXT NOT NULL REFERENCES schemes(scheme_id) ON DELETE CASCADE,
    control_id TEXT NOT NULL,
    role TEXT NOT NULL,
    weight REAL NOT NULL,
    residual_m REAL,
    vector_json TEXT NOT NULL,
    locked INTEGER NOT NULL,
    PRIMARY KEY (scheme_id, control_id)
);
CREATE TABLE IF NOT EXISTS scheme_correspondences (
    scheme_id TEXT NOT NULL REFERENCES schemes(scheme_id) ON DELETE CASCADE,
    correspondence_id TEXT NOT NULL,
    role TEXT NOT NULL,
    weight REAL NOT NULL,
    residual_m REAL,
    vector_json TEXT NOT NULL,
    group_id TEXT,
    PRIMARY KEY (scheme_id, correspondence_id)
);
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    dataset_id TEXT,
    scheme_id TEXT,
    request_json TEXT NOT NULL,
    input_fingerprint TEXT,
    rules_fingerprint TEXT,
    status TEXT NOT NULL CHECK (status IN ('succeeded','failed')),
    result_json TEXT,
    error_json TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_events (
    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    dataset_id TEXT,
    scheme_id TEXT,
    message TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    UNIQUE (job_id, event_type)
);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def connect(path: str | os.PathLike[str] | None = None) -> sqlite3.Connection:
    target = Path(path) if path is not None else DB_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(target)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    return connection


def init_db(path: str | os.PathLike[str] | None = None) -> None:
    with connect(path) as connection:
        connection.executescript(SCHEMA)


@contextmanager
def immediate_transaction(connection: sqlite3.Connection):
    connection.execute("BEGIN IMMEDIATE")
    try:
        yield connection
    except Exception:
        connection.rollback()
        raise
    else:
        connection.commit()


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def loads(value: str | None, default: Any = None) -> Any:
    return default if value is None else json.loads(value)


def one(connection: sqlite3.Connection, sql: str, parameters: Iterable[Any] = ()) -> sqlite3.Row | None:
    return connection.execute(sql, tuple(parameters)).fetchone()


def many(connection: sqlite3.Connection, sql: str, parameters: Iterable[Any] = ()) -> list[sqlite3.Row]:
    return list(connection.execute(sql, tuple(parameters)).fetchall())

"""SQLite storage.

Immutability guarantees
-----------------------
* Imported points are written once and never updated.  A failed import or
  job rolls back, so a crash never exposes a partially changed point set.
* Plans are append-only: edits create a new plan row (``parent_plan_id``),
  published plans are never rewritten.
* Job rows are state-machine guarded; audit events carry a deterministic
  ``event_key`` (job_fingerprint + event type) and a UNIQUE constraint, so
  replaying the same job cannot duplicate audit events.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, Optional

from .fingerprint import fingerprint

SCHEMA = """
CREATE TABLE IF NOT EXISTS pointsets (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    source_unit TEXT NOT NULL,
    source_crs TEXT,
    source_description TEXT,
    unit_scale_to_m REAL NOT NULL,
    created_at TEXT NOT NULL,
    input_digest TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS strips (
    id INTEGER PRIMARY KEY,
    pointset_id INTEGER NOT NULL REFERENCES pointsets(id),
    strip_uid TEXT NOT NULL,
    UNIQUE(pointset_id, strip_uid)
);
CREATE TABLE IF NOT EXISTS points (
    row_pk INTEGER PRIMARY KEY AUTOINCREMENT,
    id INTEGER NOT NULL,
    pointset_id INTEGER NOT NULL REFERENCES pointsets(id),
    strip_pk INTEGER NOT NULL REFERENCES strips(id),
    strip_uid TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    timestamp_s REAL,
    local_x REAL NOT NULL,
    local_y REAL NOT NULL,
    local_z REAL NOT NULL,
    UNIQUE(pointset_id, id)
);
CREATE INDEX IF NOT EXISTS idx_points_pid ON points(pointset_id, id);
CREATE INDEX IF NOT EXISTS idx_points_pointset ON points(pointset_id);
CREATE TABLE IF NOT EXISTS gcps (
    id INTEGER PRIMARY KEY,
    pointset_id INTEGER NOT NULL REFERENCES pointsets(id),
    point_id INTEGER NOT NULL,
    code TEXT NOT NULL,
    control_x REAL NOT NULL,
    control_y REAL NOT NULL,
    control_z REAL NOT NULL,
    control_unit TEXT NOT NULL,
    control_crs TEXT,
    UNIQUE(pointset_id, code)
);
CREATE TABLE IF NOT EXISTS correspondences (
    id INTEGER PRIMARY KEY,
    pointset_id INTEGER NOT NULL REFERENCES pointsets(id),
    point_a_id INTEGER NOT NULL,
    point_b_id INTEGER NOT NULL,
    UNIQUE(pointset_id, point_a_id, point_b_id)
);
CREATE TABLE IF NOT EXISTS plans (
    id INTEGER PRIMARY KEY,
    pointset_id INTEGER NOT NULL REFERENCES pointsets(id),
    name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    config_json TEXT NOT NULL,
    parent_plan_id INTEGER REFERENCES plans(id),
    published INTEGER NOT NULL DEFAULT 0,
    rule_version TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY,
    plan_id INTEGER NOT NULL REFERENCES plans(id),
    fingerprint TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL,
    result_json TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_plan ON jobs(plan_id);
CREATE TABLE IF NOT EXISTS audit_events (
    id INTEGER PRIMARY KEY,
    event_key TEXT NOT NULL UNIQUE,
    job_id INTEGER REFERENCES jobs(id),
    plan_id INTEGER REFERENCES plans(id),
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

VALID_JOB_TRANSITIONS = {
    "queued": {"running", "failed"},
    "running": {"succeeded", "failed"},
    "succeeded": set(),
    "failed": set(),
}

_lock = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    # Recover jobs stranded by an abnormal exit: never show partial work.
    conn.execute(
        "UPDATE jobs SET status='failed', error=COALESCE(error,'') || "
        "' [recovered: process exited while running]', updated_at=? "
        "WHERE status IN ('queued','running')",
        (utc_now(),),
    )


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    with _lock:
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except Exception:
            conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")


def dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def loads(text: Optional[str]) -> Any:
    return json.loads(text) if text else None


def emit_audit(
    conn: sqlite3.Connection,
    event_key: str,
    event_type: str,
    payload: Dict[str, Any],
    job_id: Optional[int] = None,
    plan_id: Optional[int] = None,
) -> bool:
    """Insert an audit event. Returns False if the key already exists."""
    try:
        conn.execute(
            "INSERT INTO audit_events(event_key, job_id, plan_id, event_type, "
            "payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (event_key, job_id, plan_id, event_type, dumps(payload), utc_now()),
        )
        return True
    except sqlite3.IntegrityError:
        return False


def point_input_digest(payload: Dict[str, Any]) -> str:
    return fingerprint(payload)

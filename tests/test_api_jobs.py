"""API, immutability, crash recovery, idempotent jobs and audit events."""
from __future__ import annotations

from app.demo import demo_payload


def test_import_and_demo_shape(client, demo_pointset):
    resp = client.get(f"/api/pointsets/{demo_pointset}/data")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["strips"]) == 2
    assert data["pointset"]["source_unit"] == "m"
    assert data["pointset"]["unit_scale_to_m"] == 1.0
    assert len(data["gcps"]) == 4
    # one planted bad correspondence + 4 good
    assert len(data["correspondences"]) == 5


def test_import_replay_is_idempotent(client):
    payload = demo_payload()
    r1 = client.post("/api/pointsets", json=payload).json()
    r2 = client.post("/api/pointsets", json=payload).json()
    assert r1["id"] == r2["id"]
    assert r2["replayed"] is True


def test_failed_import_leaves_no_partial_pointset(client):
    payload = demo_payload()
    payload["simulate_crash"] = True
    resp = client.post("/api/pointsets", json=payload)
    assert resp.status_code == 500
    rows = client.get("/api/audit").json()["events"]
    assert all(e["event_type"] != "plan_created" for e in rows)
    # DB must not contain the partially imported pointset.
    conn = client.app.state.db
    n = conn.execute("SELECT COUNT(*) AS c FROM pointsets").fetchone()["c"]
    assert n == 0
    npts = conn.execute("SELECT COUNT(*) AS c FROM points").fetchone()["c"]
    assert npts == 0


def test_solve_job_then_fork_and_compare(client, demo_pointset):
    ps = demo_pointset
    plan = client.post(
        f"/api/pointsets/{ps}/plans",
        json={"name": "rigid", "model_version": "rigid6-v1"},
    ).json()
    pid = plan["id"]
    job = client.post(f"/api/plans/{pid}/jobs").json()
    assert job["replayed"] is False
    assert job["job"]["status"] == "succeeded"
    result = job["job"]["result"]
    assert result["status"] == "ok"
    assert result["rank"] == result["n_params"]
    # The planted outlier correspondence is rejected but retained.
    accepted = {c["corr_id"]: c["accepted"] for c in result["corr_residuals"]}
    assert any(v is False for v in accepted.values())

    # Replay the same job: no duplicate audit event.
    audit_before = client.get("/api/audit").json()["events"]
    replay = client.post(f"/api/plans/{pid}/jobs").json()
    assert replay["replayed"] is True
    assert replay["job"]["id"] == job["job"]["id"]
    audit_after = client.get("/api/audit").json()["events"]
    assert len(audit_before) == len(audit_after)

    # Fork disables the rejected correspondence explicitly.
    rejected = result["rejected_corr_ids"]
    forked = client.post(
        f"/api/plans/{pid}/fork",
        json={"name": "rigid no-outlier",
              "changes": {"disabled_corr_ids": rejected}},
    ).json()
    assert forked["parent_plan_id"] == pid
    fid = forked["id"]
    client.post(f"/api/plans/{fid}/jobs")
    comp = client.get(f"/api/plans/{pid}/compare/{fid}").json()
    assert comp["plan_a"]["plan"]["id"] == pid
    assert comp["plan_b"]["plan"]["id"] == fid
    assert "points" in comp["plan_a"] and "points" in comp["plan_b"]


def test_published_plan_cannot_be_modified_in_place(client, demo_pointset):
    ps = demo_pointset
    plan = client.post(
        f"/api/pointsets/{ps}/plans", json={"name": "p", "model_version": "rigid6-v1"}
    ).json()
    pid = plan["id"]
    client.post(f"/api/plans/{pid}/publish")
    # Publishing twice is a no-op, not a rewrite.
    client.post(f"/api/plans/{pid}/publish")
    after = client.get(f"/api/plans/{pid}").json()
    assert after["published"] == 1
    # There is no update endpoint; all changes go through /fork.
    assert client.put(f"/api/plans/{pid}", json={}).status_code == 405


def test_crash_job_is_recovered_as_failed_and_can_rerun(client, demo_pointset):
    ps = demo_pointset
    plan = client.post(
        f"/api/pointsets/{ps}/plans", json={"name": "p", "model_version": "rigid6-v1"}
    ).json()
    pid = plan["id"]
    resp = client.post(f"/api/plans/{pid}/jobs", json={"simulate_crash": True})
    assert resp.status_code == 500
    conn = client.app.state.db
    row = conn.execute(
        "SELECT status, error FROM jobs WHERE plan_id=?", (pid,)
    ).fetchone()
    assert row["status"] == "failed"
    assert "simulated crash" in row["error"]
    # A fresh normal run succeeds (different fingerprint is not required;
    # the failed job's fingerprint was consumed as failed, so rerun with the
    # same payload returns the failed row idempotently instead of duplicating).
    # Normal rerun path is exercised through a forked plan.
    forked = client.post(
        f"/api/plans/{pid}/fork", json={"name": "rerun", "changes": {"outlier_sigma": 1.0}}
    ).json()
    ok = client.post(f"/api/plans/{forked['id']}/jobs").json()
    assert ok["job"]["status"] == "succeeded"


def test_startup_recovers_stranded_running_job(client, demo_pointset):
    from app.db import connect, init_db
    ps = demo_pointset
    plan = client.post(
        f"/api/pointsets/{ps}/plans", json={"name": "p", "model_version": "rigid6-v1"}
    ).json()
    pid = plan["id"]
    db_path = client.app.state.db.execute("PRAGMA database_list").fetchone()[2]
    conn = client.app.state.db
    now = "2026-09-20T00:00:00.000000Z"
    conn.execute(
        "INSERT INTO jobs(plan_id, fingerprint, status, created_at, updated_at) "
        "VALUES (?, ?, 'running', ?, ?)",
        (pid, "stranded-fingerprint-xyz", now, now),
    )
    conn.commit()
    # Simulate process restart: a fresh connection runs schema/recovery.
    fresh = connect(db_path)
    init_db(fresh)
    row = fresh.execute(
        "SELECT status, error FROM jobs WHERE fingerprint='stranded-fingerprint-xyz'"
    ).fetchone()
    assert row["status"] == "failed"
    assert "recovered" in row["error"]
    fresh.close()


def test_fork_disabling_correspondence_group_runs(client, demo_pointset):
    ps = demo_pointset
    plan = client.post(
        f"/api/pointsets/{ps}/plans",
        json={"name": "rigid", "model_version": "rigid6-v1"},
    ).json()
    # Disable the only overlap group and the worst gcps to reach a clean
    # rank-deficient solve (no correspondence observations remain).
    forked = client.post(
        f"/api/plans/{plan['id']}/fork",
        json={"name": "no-group", "changes": {
            "disabled_corr_groups": ["S1:S2"],
            "disabled_gcp_ids": [3, 4],
        }},
    ).json()
    assert forked["config"]["disabled_corr_groups"] == ["S1:S2"]
    job = client.post(f"/api/plans/{forked['id']}/jobs").json()["job"]
    assert job["result"]["status"] == "rank_deficient"
    assert job["result"]["rank"] < job["result"]["n_params"]
    assert len(job["result"]["candidates"]) >= 1

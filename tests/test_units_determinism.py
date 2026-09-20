"""Unit provenance and deterministic re-solve."""
from __future__ import annotations

import numpy as np

from app.demo import demo_payload


def test_unknown_unit_rejected(client, tmp_db):
    payload = demo_payload()
    payload["unit"] = "furlong"
    resp = client.post("/api/pointsets", json=payload)
    assert resp.status_code == 400
    assert "unknown unit" in resp.json()["detail"]


def test_cm_import_normalizes_but_keeps_source(client):
    payload = demo_payload()
    # Convert every local coordinate from metres to centimetres.
    for strip in payload["strips"]:
        for point in strip["points"]:
            point["x"] *= 100.0
            point["y"] *= 100.0
            point["z"] *= 100.0
    payload["unit"] = "cm"
    ps = client.post("/api/pointsets", json=payload).json()["id"]
    data = client.get(f"/api/pointsets/{ps}/data").json()
    pointset = data["pointset"]
    assert pointset["source_unit"] == "cm"
    assert abs(pointset["unit_scale_to_m"] - 0.01) < 1e-12
    # Normalized values must match the metre demo exactly.
    s1 = next(st for st in data["strips"] if st["strip_uid"] == "S1")
    first_cm = np.array(s1["points"][0]["xyz"])
    demo_first = demo_payload()["strips"][0]["points"][0]
    np.testing.assert_allclose(
        first_cm, [demo_first["x"], demo_first["y"], demo_first["z"]], atol=1e-9
    )


def test_repeated_solves_are_bitwise_deterministic(client, demo_pointset):
    ps = demo_pointset
    results = []
    for _ in range(2):
        plan = client.post(
            f"/api/pointsets/{ps}/plans",
            json={"name": f"rigid-{_}", "model_version": "strip_offset3-v1"},
        ).json()
        job = client.post(f"/api/plans/{plan['id']}/jobs").json()["job"]
        results.append(job["result"])
    a, b = results
    assert a["params"] == b["params"]
    assert a["singular_values"] == b["singular_values"]
    assert a["rank"] == b["rank"]
    assert [c["norm_m"] for c in a["corr_residuals"]] == \
           [c["norm_m"] for c in b["corr_residuals"]]


def test_demo_offset_model_recovers_known_shifts(client, demo_pointset):
    from app.demo import SHIFT1, SHIFT2
    ps = demo_pointset
    plan = client.post(
        f"/api/pointsets/{ps}/plans",
        json={"name": "offset", "model_version": "strip_offset3-v1"},
    ).json()
    result = client.post(f"/api/plans/{plan['id']}/jobs").json()["job"]["result"]
    assert result["status"] == "ok"
    params = np.array(result["params"]).reshape(2, 3)
    np.testing.assert_allclose(params[0], SHIFT1, atol=1e-6)
    np.testing.assert_allclose(params[1], SHIFT2, atol=1e-6)
    # Only the single planted gross outlier is rejected; 4 good corrs remain.
    assert len(result["rejected_corr_ids"]) == 1
    accepted = [c for c in result["corr_residuals"] if c["accepted"]]
    assert len(accepted) == 4
    assert all(c["norm_m"] < 1e-6 for c in accepted)


def test_corrected_view_uses_local_to_control_direction(client, demo_pointset):
    ps = demo_pointset
    plan = client.post(
        f"/api/pointsets/{ps}/plans",
        json={"name": "offset", "model_version": "strip_offset3-v1"},
    ).json()
    client.post(f"/api/plans/{plan['id']}/jobs")
    view = client.get(f"/api/plans/{plan['id']}/corrected").json()
    assert view["unit"] == "m"
    # corrected - local should equal the recovered strip translation.
    s1 = view["strips"]["S1"]
    p0 = s1[0]
    delta = np.array(p0["corrected_xyz_m"]) - np.array(p0["local_xyz_m"])
    assert abs(delta[0] - 0.30) < 1e-6
    assert abs(delta[1] - (-0.15)) < 1e-6
    assert abs(delta[2] - 0.08) < 1e-6

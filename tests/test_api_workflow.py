def import_payload():
    return {
        "job_id": "job-import-1",
        "name": "API test dataset",
        "source_name": "test-extract",
        "common_crs": "test common engineering CRS",
        "common_length_unit": "m",
        "metadata": {"source_coordinates": "verbatim"},
        "strips": [
            {"strip_id": "A", "time_start": 0, "time_end": 10, "local_crs": "A local", "local_length_unit": "m"},
            {"strip_id": "B", "time_start": 0, "time_end": 10, "local_crs": "B local", "local_length_unit": "m"},
        ],
        "points": [
            {"point_id": "a0", "strip_id": "A", "time_value": 0, "x": 0, "y": 0, "z": 0},
            {"point_id": "a1", "strip_id": "A", "time_value": 10, "x": 10, "y": 0, "z": 1},
            {"point_id": "a2", "strip_id": "A", "time_value": 5, "x": 5, "y": 4, "z": 1},
            {"point_id": "b0", "strip_id": "B", "time_value": 0, "x": 0, "y": 0, "z": 0},
            {"point_id": "b1", "strip_id": "B", "time_value": 10, "x": 10, "y": 0, "z": 1},
            {"point_id": "b2", "strip_id": "B", "time_value": 5, "x": 5, "y": 4, "z": 1},
        ],
        "controls": [
            {"control_id": "g0", "point_id": "a0", "x": 0, "y": 0, "z": 0, "weight": 10},
            {"control_id": "g1", "point_id": "a1", "x": 10, "y": 0, "z": 1, "weight": 10},
            {"control_id": "g2", "point_id": "a2", "x": 5, "y": 4, "z": 1, "weight": 10},
        ],
        "correspondences": [
            {"correspondence_id": "good", "left_point_id": "a0", "right_point_id": "b0", "weight": 1, "group_id": "g"},
            {"correspondence_id": "good2", "left_point_id": "a2", "right_point_id": "b2", "weight": 1, "group_id": "g"},
            {"correspondence_id": "bad", "left_point_id": "a1", "right_point_id": "b0", "weight": 0.1, "group_id": "bad-group"},
        ],
    }


def test_import_replay_is_idempotent_and_no_partial_points_on_failure(client):
    response = client.post("/api/datasets/import", json=import_payload())
    assert response.status_code == 200
    dataset_id = response.json()["dataset_id"]

    replay = client.post("/api/datasets/import", json=import_payload())
    assert replay.status_code == 200
    assert replay.json()["replayed"] is True
    job = client.get("/api/jobs/job-import-1").json()
    assert len(job["audit_events"]) == 1
    assert job["audit_events"][0]["event_type"] == "dataset_imported"

    bad = import_payload()
    bad["job_id"] = "job-import-bad"
    bad["points"][0]["x"] = "not-a-number"
    failed = client.post("/api/datasets/import", json=bad)
    assert failed.status_code == 422
    listing = client.get("/api/datasets").json()["datasets"]
    assert len(listing) == 1
    assert listing[0]["dataset_id"] == dataset_id


def test_solve_rejects_over_limit_correspondence_but_retains_evidence(client):
    dataset_id = client.post("/api/datasets/import", json=import_payload()).json()["dataset_id"]
    payload = {
        "job_id": "job-solve-1", "dataset_id": dataset_id, "label": "offset with rejection",
        "model": "strip_offset_v1", "residual_limit_m": 1.0,
        "disabled_control_ids": [], "locked_control_ids": [],
        "disabled_correspondence_ids": [], "disabled_correspondence_group_ids": [],
        "weights": {}, "segment_splits": {},
    }
    result = client.post("/api/schemes/solve", json=payload).json()
    assert result["scheme_id"]
    detail = client.get(f"/api/schemes/{result['scheme_id']}").json()
    residuals = detail["scheme"]["result"]["residuals"]
    assert "bad" in residuals
    assert residuals["bad"]["status"] == "rejected"
    assert residuals["bad"]["norm"] > 1.0
    assert {row["correspondence_id"]: row["role"] for row in detail["evidence_correspondences"]}["bad"] == "rejected"


def test_rerun_same_job_does_not_duplicate_scheme_audit(client):
    dataset_id = client.post("/api/datasets/import", json=import_payload()).json()["dataset_id"]
    payload = {"job_id": "job-solve-x", "dataset_id": dataset_id, "label": "rigid", "model": "rigid_v1"}
    first = client.post("/api/schemes/solve", json=payload).json()
    second = client.post("/api/schemes/solve", json=payload).json()
    assert first["scheme_id"] == second["scheme_id"]
    assert second["replayed"] is True
    events = client.get("/api/jobs/job-solve-x").json()["audit_events"]
    assert len(events) == 1


def test_degenerate_solve_returns_rank_nullspace_and_creates_no_scheme(client):
    dataset_id = client.post("/api/datasets/import", json=import_payload()).json()["dataset_id"]
    payload = {
        "job_id": "job-solve-degenerate", "dataset_id": dataset_id, "label": "degenerate",
        "model": "strip_offset_v1",
        "disabled_control_ids": ["g0", "g1", "g2"],
    }
    result = client.post("/api/schemes/solve", json=payload).json()
    assert result["scheme_id"] is None
    assert result["error"]["error_code"] == "rank_deficient"
    assert result["error"]["rank"] < result["error"]["degrees_of_freedom"]
    assert result["error"]["unconstrained_directions"]
    assert len(result["error"]["candidates"]) > 1
    job = client.get("/api/jobs/job-solve-degenerate").json()["job"]
    assert job["status"] == "failed"
    assert client.get(f"/api/datasets/{dataset_id}/schemes").json()["schemes"] == []


def test_source_coordinates_are_not_replaced_by_transformed_points(client):
    dataset_id = client.post("/api/datasets/import", json=import_payload()).json()["dataset_id"]
    solve = {"job_id": "job-solve-publish", "dataset_id": dataset_id, "label": "rigid", "model": "rigid_v1"}
    scheme_id = client.post("/api/schemes/solve", json=solve).json()["scheme_id"]
    publish = client.post("/api/schemes/publish", json={"job_id": "job-publish-1", "scheme_id": scheme_id})
    assert publish.status_code == 200
    source = client.get(f"/api/datasets/{dataset_id}").json()
    point_a0 = next(point for point in source["points"] if point["point_id"] == "a0")
    assert [point_a0["x"], point_a0["y"], point_a0["z"]] == [0.0, 0.0, 0.0]
    detail = client.get(f"/api/schemes/{scheme_id}").json()
    transformed = next(point for point in detail["transformed_points"] if point["point_id"] == "a0")
    assert "common" in transformed


def test_unknown_control_reference_is_structured_422(client):
    dataset_id = client.post("/api/datasets/import", json=import_payload()).json()["dataset_id"]
    payload = {
        "job_id": "job-unknown-control", "dataset_id": dataset_id, "label": "bad reference",
        "model": "rigid_v1", "locked_control_ids": ["missing"],
    }
    response = client.post("/api/schemes/solve", json=payload)
    assert response.status_code == 422
    assert response.json()["error_code"] == "unknown_control_id"
    assert client.get("/api/jobs/job-unknown-control").status_code == 404


def test_rigid_model_rejects_segment_split(client):
    dataset_id = client.post("/api/datasets/import", json=import_payload()).json()["dataset_id"]
    payload = {
        "job_id": "job-rigid-split", "dataset_id": dataset_id, "label": "bad split",
        "model": "rigid_v1", "segment_splits": {"A": 5.0},
    }
    response = client.post("/api/schemes/solve", json=payload)
    assert response.status_code == 422
    assert response.json()["error_code"] == "segments_not_supported"


def test_two_schemes_for_same_dataset_can_be_compared(client):
    dataset_id = client.post("/api/datasets/import", json=import_payload()).json()["dataset_id"]
    rigid = client.post("/api/schemes/solve", json={
        "job_id": "job-compare-rigid", "dataset_id": dataset_id, "label": "rigid", "model": "rigid_v1",
    }).json()["scheme_id"]
    offset = client.post("/api/schemes/solve", json={
        "job_id": "job-compare-offset", "dataset_id": dataset_id, "label": "offset", "model": "strip_offset_v1",
    }).json()["scheme_id"]
    response = client.get(f"/api/compare/{rigid}/{offset}")
    assert response.status_code == 200
    body = response.json()
    assert body["left"]["scheme"]["scheme_id"] == rigid
    assert body["right"]["scheme"]["scheme_id"] == offset


def test_failed_rank_deficient_job_replays_without_another_audit_event(client):
    dataset_id = client.post("/api/datasets/import", json=import_payload()).json()["dataset_id"]
    payload = {
        "job_id": "job-failed-replay", "dataset_id": dataset_id, "label": "degenerate",
        "model": "strip_offset_v1", "disabled_control_ids": ["g0", "g1", "g2"],
    }
    first = client.post("/api/schemes/solve", json=payload).json()
    second = client.post("/api/schemes/solve", json=payload).json()
    assert first["scheme_id"] is None
    assert second["replayed"] is True
    assert second["scheme_id"] is None
    assert second["error"]["error_code"] == "rank_deficient"
    events = client.get("/api/jobs/job-failed-replay").json()["audit_events"]
    assert len(events) == 1


def test_mixed_units_are_imported_verbatim_but_not_solved_together(client):
    payload = import_payload()
    payload["job_id"] = "job-import-mixed-units"
    payload["strips"][1]["local_length_unit"] = "ft"
    imported = client.post("/api/datasets/import", json=payload)
    assert imported.status_code == 200
    dataset_id = imported.json()["dataset_id"]
    detail = client.get(f"/api/datasets/{dataset_id}").json()
    assert [strip["local_length_unit"] for strip in detail["strips"]] == ["m", "ft"]
    solve = client.post("/api/schemes/solve", json={
        "job_id": "job-solve-mixed-units", "dataset_id": dataset_id, "label": "mixed", "model": "rigid_v1",
    })
    assert solve.status_code == 422
    assert solve.json()["error_code"] == "mixed_length_units"

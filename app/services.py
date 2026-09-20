from __future__ import annotations

import hashlib
import sqlite3
from typing import Any

from .adjustment import ControlRecord, CorrespondenceRecord, PointRecord, solve_adjustment
from .database import dumps, immediate_transaction, loads, many, now_iso, one
from .fingerprint import canonical_json, fingerprint


class ServiceError(ValueError):
    def __init__(self, status_code: int, error_code: str, message: str, details: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.details = details


def stable_id(prefix: str, *parts: Any) -> str:
    return prefix + "_" + hashlib.sha256(canonical_json(parts).encode("utf-8")).hexdigest()[:24]


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ServiceError(422, "invalid_text", f"{name} is required")
    return value


def _number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ServiceError(422, "invalid_number", f"{name} must be numeric") from exc
    if number in (float("inf"), float("-inf")) or number != number:
        raise ServiceError(422, "invalid_number", f"{name} must be finite")
    return number


def get_existing_job(connection: sqlite3.Connection, job_id: str) -> dict[str, Any] | None:
    row = one(connection, "SELECT * FROM jobs WHERE job_id = ?", (job_id,))
    if row is None:
        return None
    data = dict(row)
    data["request"] = loads(data.pop("request_json"))
    data["result"] = loads(data.pop("result_json"), None)
    data["error"] = loads(data.pop("error_json"), None)
    return data


def _finish_job(connection, job_id, kind, dataset_id, scheme_id, request, input_fp, rules_fp, status, result=None, error=None):
    connection.execute(
        """INSERT INTO jobs
        (job_id,kind,dataset_id,scheme_id,request_json,input_fingerprint,rules_fingerprint,
         status,result_json,error_json,created_at,completed_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (job_id, kind, dataset_id, scheme_id, dumps(request), input_fp, rules_fp,
         status, dumps(result) if result is not None else None,
         dumps(error) if error is not None else None, now_iso(), now_iso()),
    )


def _audit(connection, job_id, event_type, dataset_id=None, scheme_id=None, message=""):
    connection.execute(
        "INSERT INTO audit_events (job_id,event_type,dataset_id,scheme_id,message,occurred_at) VALUES (?,?,?,?,?,?)",
        (job_id, event_type, dataset_id, scheme_id, message, now_iso()),
    )


def _replay(existing, kind, fp):
    if existing["kind"] != kind or existing["input_fingerprint"] != fp:
        raise ServiceError(409, "job_id_reused", "job_id was already used with a different operation")
    return {
        "replayed": True,
        "job": existing,
        "dataset_id": existing["dataset_id"],
        "scheme_id": existing["scheme_id"],
        "error": existing.get("error"),
    }


def import_dataset(connection: sqlite3.Connection, request: dict[str, Any]) -> dict[str, Any]:
    job_id = request["job_id"]
    source_fp = fingerprint(request)
    existing = get_existing_job(connection, job_id)
    if existing:
        return _replay(existing, "import", source_fp)
    dataset_id = stable_id("dataset", source_fp)
    with immediate_transaction(connection):
        try:
            if one(connection, "SELECT 1 FROM datasets WHERE dataset_id=?", (dataset_id,)):
                raise ServiceError(409, "dataset_exists", "identical dataset already exists; choose a new job_id")
            _insert_dataset(connection, request, dataset_id, source_fp)
            _finish_job(connection, job_id, "import", dataset_id, None, request, source_fp, source_fp, "succeeded", {"dataset_id": dataset_id})
            _audit(connection, job_id, "dataset_imported", dataset_id=dataset_id, message=request["name"])
        except ServiceError:
            raise
    return {"replayed": False, "job": get_existing_job(connection, job_id), "dataset_id": dataset_id}


def _insert_dataset(connection, request, dataset_id, source_fp):
    name = _text(request.get("name"), "name")
    source_name = _text(request.get("source_name"), "source_name")
    common_crs = _text(request.get("common_crs"), "common_crs")
    common_unit = request.get("common_length_unit")
    if common_unit not in ("m", "ft", "survey_ft"):
        raise ServiceError(422, "invalid_unit", "common_length_unit must be m, ft, or survey_ft")
    strips = request.get("strips")
    points = request.get("points")
    if not isinstance(strips, list) or not strips or not isinstance(points, list) or not points:
        raise ServiceError(422, "missing_features", "strips and points are required")

    strip_map = {}
    for raw in strips:
        strip_id = _text(raw.get("strip_id"), "strip.strip_id")
        if strip_id in strip_map:
            raise ServiceError(422, "duplicate_strip", f"duplicate strip {strip_id}")
        start = _number(raw.get("time_start"), f"strip {strip_id}.time_start")
        end = _number(raw.get("time_end"), f"strip {strip_id}.time_end")
        if end <= start:
            raise ServiceError(422, "invalid_strip_time", f"strip {strip_id} must have end > start")
        strip_map[strip_id] = {
            "strip_id": strip_id,
            "label": _text(raw.get("label", strip_id), "strip.label"),
            "time_start": start,
            "time_end": end,
            "time_unit": _text(raw.get("time_unit", "s"), "time_unit"),
            "time_epoch": _text(raw.get("time_epoch", "source_clock"), "time_epoch"),
            "local_crs": _text(raw.get("local_crs"), f"strip {strip_id}.local_crs"),
            "local_length_unit": raw.get("local_length_unit", common_unit),
            "metadata": raw.get("metadata", {}),
        }
        # Different source units are preserved verbatim.  The solver rejects a
        # mixed model instead of performing an undeclared unit conversion.
        if not isinstance(strip_map[strip_id]["metadata"], dict):
            raise ServiceError(422, "invalid_metadata", f"strip {strip_id} metadata must be an object")

    point_map = {}
    for raw in points:
        point_id = _text(raw.get("point_id"), "point.point_id")
        if point_id in point_map:
            raise ServiceError(422, "duplicate_point", f"duplicate point {point_id}")
        strip_id = _text(raw.get("strip_id"), "point.strip_id")
        if strip_id not in strip_map:
            raise ServiceError(422, "unknown_strip", f"point {point_id} references unknown strip")
        time_value = _number(raw.get("time_value"), f"point {point_id}.time_value")
        strip = strip_map[strip_id]
        if not strip["time_start"] <= time_value <= strip["time_end"]:
            raise ServiceError(422, "point_outside_strip_time", f"point {point_id} is outside strip time range")
        point_map[point_id] = {"point_id": point_id, "strip_id": strip_id, "time_value": time_value,
                               "x": _number(raw.get("x"), "x"), "y": _number(raw.get("y"), "y"), "z": _number(raw.get("z"), "z")}

    controls = []
    seen_controls = set()
    for raw in request.get("controls") or []:
        control_id = _text(raw.get("control_id"), "control.control_id")
        if control_id in seen_controls:
            raise ServiceError(422, "duplicate_control", f"duplicate control {control_id}")
        seen_controls.add(control_id)
        point_id = _text(raw.get("point_id"), "control.point_id")
        if point_id not in point_map:
            raise ServiceError(422, "unknown_control_point", f"control {control_id} point not found")
        weight = _number(raw.get("weight", 1.0), "control.weight")
        if weight <= 0:
            raise ServiceError(422, "invalid_weight", "weights must be positive")
        controls.append((control_id, point_id, weight, raw))

    correspondences = []
    seen_corr = set()
    for raw in request.get("correspondences") or []:
        cid = _text(raw.get("correspondence_id"), "correspondence.correspondence_id")
        if cid in seen_corr:
            raise ServiceError(422, "duplicate_correspondence", f"duplicate correspondence {cid}")
        seen_corr.add(cid)
        left = _text(raw.get("left_point_id"), "left_point_id")
        right = _text(raw.get("right_point_id"), "right_point_id")
        if left not in point_map or right not in point_map:
            raise ServiceError(422, "unknown_correspondence_point", f"correspondence {cid} point not found")
        if point_map[left]["strip_id"] == point_map[right]["strip_id"]:
            raise ServiceError(422, "correspondence_same_strip", f"correspondence {cid} must connect different strips")
        weight = _number(raw.get("weight", 1.0), "correspondence.weight")
        if weight <= 0:
            raise ServiceError(422, "invalid_weight", "weights must be positive")
        correspondences.append((cid, left, right, weight, raw.get("group_id"), raw.get("metadata", {})))

    connection.execute(
        """INSERT INTO datasets
        (dataset_id,name,source_name,common_crs,common_length_unit,metadata_json,input_fingerprint,created_job_id,created_at)
        VALUES (?,?,?,?,?,?,?,?,?)""",
        (dataset_id, name, source_name, common_crs, common_unit, dumps(request.get("metadata", {})),
         source_fp, request["job_id"], now_iso()),
    )
    for strip in strip_map.values():
        connection.execute(
            """INSERT INTO strips (dataset_id,strip_id,label,time_start,time_end,time_unit,time_epoch,
            local_crs,local_length_unit,metadata_json) VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (dataset_id, strip["strip_id"], strip["label"], strip["time_start"], strip["time_end"],
             strip["time_unit"], strip["time_epoch"], strip["local_crs"], strip["local_length_unit"],
             dumps(strip["metadata"])),
        )
    for point in point_map.values():
        connection.execute(
            "INSERT INTO points (dataset_id,point_id,strip_id,time_value,x,y,z) VALUES (?,?,?,?,?,?,?)",
            (dataset_id, point["point_id"], point["strip_id"], point["time_value"], point["x"], point["y"], point["z"]),
        )
    for control_id, point_id, weight, raw in controls:
        connection.execute(
            "INSERT INTO controls (dataset_id,control_id,point_id,x,y,z,weight,metadata_json) VALUES (?,?,?,?,?,?,?,?)",
            (dataset_id, control_id, point_id, _number(raw.get("x"), "control x"),
             _number(raw.get("y"), "control y"), _number(raw.get("z"), "control z"), weight,
             dumps(raw.get("metadata", {}))),
        )
    for cid, left, right, weight, group_id, meta in correspondences:
        connection.execute(
            """INSERT INTO correspondences
            (dataset_id,correspondence_id,left_point_id,right_point_id,weight,group_id,metadata_json)
            VALUES (?,?,?,?,?,?,?)""",
            (dataset_id, cid, left, right, weight, group_id, dumps(meta)),
        )


def _load_dataset(connection, dataset_id):
    dataset = one(connection, "SELECT * FROM datasets WHERE dataset_id=?", (dataset_id,))
    if dataset is None:
        raise ServiceError(404, "dataset_not_found", "dataset not found")
    strips = many(connection, "SELECT * FROM strips WHERE dataset_id=? ORDER BY strip_id", (dataset_id,))
    points = many(connection, "SELECT * FROM points WHERE dataset_id=? ORDER BY point_id", (dataset_id,))
    controls = many(connection, "SELECT * FROM controls WHERE dataset_id=? ORDER BY control_id", (dataset_id,))
    correspondences = many(connection, "SELECT * FROM correspondences WHERE dataset_id=? ORDER BY correspondence_id", (dataset_id,))
    return dict(dataset), [dict(row) for row in strips], [dict(row) for row in points], [dict(row) for row in controls], [dict(row) for row in correspondences]



def _validate_solve_request(strip_rows, control_rows, corr_rows, request):
    model = request.get("model")
    if model not in ("rigid_v1", "strip_offset_v1", "drift_v1"):
        raise ServiceError(422, "unknown_model", "unknown model version")
    control_ids = {row["control_id"] for row in control_rows}
    corr_ids = {row["correspondence_id"] for row in corr_rows}
    group_ids = {row["group_id"] for row in corr_rows if row["group_id"] is not None}
    strip_ids = {row["strip_id"] for row in strip_rows}

    def identifier_list(key):
        value = request.get(key, [])
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ServiceError(422, "invalid_identifier_list", f"{key} must be a list of strings")
        return value

    disabled_controls = set(identifier_list("disabled_control_ids"))
    locked_controls = set(identifier_list("locked_control_ids"))
    disabled_corrs = set(identifier_list("disabled_correspondence_ids"))
    disabled_groups = set(identifier_list("disabled_correspondence_group_ids"))
    unknown_controls = (disabled_controls | locked_controls) - control_ids
    if unknown_controls:
        raise ServiceError(422, "unknown_control_id", f"unknown control {sorted(unknown_controls)[0]}")
    if disabled_controls & locked_controls:
        raise ServiceError(422, "control_conflict", "a control cannot be disabled and locked")
    unknown_corrs = disabled_corrs - corr_ids
    if unknown_corrs:
        raise ServiceError(422, "unknown_correspondence_id", f"unknown correspondence {sorted(unknown_corrs)[0]}")
    unknown_groups = disabled_groups - group_ids
    if unknown_groups:
        raise ServiceError(422, "unknown_correspondence_group", f"unknown correspondence group {sorted(unknown_groups)[0]}")

    weights = request.get("weights", {})
    if not isinstance(weights, dict):
        raise ServiceError(422, "invalid_weights", "weights must be an object")
    for key, weight in weights.items():
        try:
            numeric = float(weight)
        except (TypeError, ValueError) as exc:
            raise ServiceError(422, "invalid_weight", f"weight {key} must be numeric") from exc
        if not numeric == numeric or numeric in (float("inf"), float("-inf")) or numeric <= 0.0:
            raise ServiceError(422, "invalid_weight", f"weight {key} must be finite and positive")
        if key.startswith("control:"):
            if key.split(":", 1)[1] not in control_ids:
                raise ServiceError(422, "unknown_weight_target", f"unknown control in weight {key}")
        elif key.startswith("correspondence:"):
            if key.split(":", 1)[1] not in corr_ids:
                raise ServiceError(422, "unknown_weight_target", f"unknown correspondence in weight {key}")
        else:
            raise ServiceError(422, "unknown_weight_target", "weight keys must start with control: or correspondence:")

    splits = request.get("segment_splits", {})
    if not isinstance(splits, dict):
        raise ServiceError(422, "invalid_segment_splits", "segment_splits must be an object")
    if model == "rigid_v1" and splits:
        raise ServiceError(422, "segments_not_supported", "rigid_v1 cannot be split into correction segments")
    strip_by_id = {row["strip_id"]: row for row in strip_rows}
    for strip_id, raw_split in splits.items():
        if strip_id not in strip_ids:
            raise ServiceError(422, "unknown_strip_split", f"unknown strip {strip_id} in segment_splits")
        try:
            split = float(raw_split)
        except (TypeError, ValueError) as exc:
            raise ServiceError(422, "invalid_strip_split", f"split for {strip_id} must be numeric") from exc
        strip = strip_by_id[strip_id]
        if not strip["time_start"] < split < strip["time_end"]:
            raise ServiceError(422, "invalid_strip_split", f"split for {strip_id} must be strictly inside its time range")

    limit = request.get("residual_limit_m")
    if limit is not None:
        try:
            numeric_limit = float(limit)
        except (TypeError, ValueError) as exc:
            raise ServiceError(422, "invalid_residual_limit", "residual_limit_m must be numeric") from exc
        if not numeric_limit == numeric_limit or numeric_limit in (float("inf"), float("-inf")) or numeric_limit <= 0.0:
            raise ServiceError(422, "invalid_residual_limit", "residual_limit_m must be finite and positive")
    iterations = request.get("max_iterations", 20)
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations <= 0:
        raise ServiceError(422, "invalid_max_iterations", "max_iterations must be a positive integer")
    try:
        tolerance = float(request.get("step_tolerance", 1e-11))
    except (TypeError, ValueError) as exc:
        raise ServiceError(422, "invalid_step_tolerance", "step_tolerance must be numeric") from exc
    if not tolerance == tolerance or tolerance in (float("inf"), float("-inf")) or tolerance <= 0.0:
        raise ServiceError(422, "invalid_step_tolerance", "step_tolerance must be finite and positive")


def _solve_payload(dataset, strips, point_rows, control_rows, corr_rows, request):
    point_records = {
        row["point_id"]: PointRecord(row["point_id"], row["strip_id"], row["time_value"], (row["x"], row["y"], row["z"]))
        for row in point_rows
    }
    controls = [ControlRecord(row["control_id"], row["point_id"], (row["x"], row["y"], row["z"]), row["weight"]) for row in control_rows]
    correspondences = [
        CorrespondenceRecord(row["correspondence_id"], row["left_point_id"], row["right_point_id"], row["weight"], row["group_id"])
        for row in corr_rows
    ]
    defaults = {
        "disabled_control_ids": [], "locked_control_ids": [],
        "disabled_correspondence_ids": [], "disabled_correspondence_group_ids": [],
        "weights": {}, "segment_splits": {}, "residual_limit_m": None,
        "max_iterations": 20, "step_tolerance": 1e-11,
    }
    config = {key: request.get(key, defaults[key]) for key in defaults}
    result = solve_adjustment(request["model"], strips, point_records, controls, correspondences, config)
    return result, config


def _rules_fingerprint(dataset_id, dataset_fp, model, config, parent_scheme_id):
    rules = {
        "dataset_id": dataset_id,
        "dataset_input_fingerprint": dataset_fp,
        "model_version": model,
        "config": config,
        "parent_scheme_id": parent_scheme_id,
        "solver_version": "deterministic_gn_v1",
    }
    return fingerprint(rules), rules


def solve_scheme(connection: sqlite3.Connection, request: dict[str, Any]) -> dict[str, Any]:
    job_id = request["job_id"]
    input_fp = fingerprint(request)
    existing = get_existing_job(connection, job_id)
    if existing:
        return _replay(existing, "solve", input_fp)
    dataset, strips, point_rows, control_rows, corr_rows = _load_dataset(connection, request["dataset_id"])
    parent = request.get("parent_scheme_id")
    if parent is not None:
        parent_row = one(connection, "SELECT scheme_id FROM schemes WHERE scheme_id=? AND dataset_id=?", (parent, request["dataset_id"]))
        if parent_row is None:
            raise ServiceError(404, "parent_scheme_not_found", "parent scheme not found in this dataset")
    _validate_solve_request(strips, control_rows, corr_rows, request)
    strip_units = {row["local_length_unit"] for row in strips}
    if strip_units != {dataset["common_length_unit"]}:
        raise ServiceError(422, "mixed_length_units", "source units are preserved; mixed-unit adjustment requires an explicit conversion outside this application")
    result, config = _solve_payload(dataset, strips, point_rows, control_rows, corr_rows, request)
    rules_fp, rules = _rules_fingerprint(request["dataset_id"], dataset["input_fingerprint"], request["model"], config, parent)
    scheme_id = stable_id("scheme", rules_fp)
    with immediate_transaction(connection):
        if one(connection, "SELECT 1 FROM schemes WHERE scheme_id=?", (scheme_id,)):
            raise ServiceError(409, "scheme_exists", "identical scheme already exists; choose a new job_id")
        if result["status"] != "ok":
            error = {
                "error_code": result.get("error_code"),
                "message": result.get("message", "adjustment is rank-deficient"),
                "rank": result.get("rank"),
                "degrees_of_freedom": result.get("degrees_of_freedom"),
                "unconstrained_directions": result.get("unconstrained_directions", []),
                "candidates": result.get("candidates", []),
                "rejected_correspondence_ids": result.get("rejected_correspondence_ids", []),
            }
            _finish_job(connection, job_id, "solve", request["dataset_id"], None, request, input_fp, rules_fp, "failed", None, error)
            _audit(connection, job_id, "solve_rejected", dataset_id=request["dataset_id"], message=error["message"])
            return {"replayed": False, "job": get_existing_job(connection, job_id), "scheme_id": None, "error": error}
        connection.execute(
            """INSERT INTO schemes
            (scheme_id,dataset_id,parent_scheme_id,label,model_version,status,config_json,rules_fingerprint,
             result_json,created_job_id,created_at,published_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,NULL)""",
            (scheme_id, request["dataset_id"], parent, request["label"], request["model"], "draft",
             dumps(rules), rules_fp, dumps(result), job_id, now_iso()),
        )
        for control in control_rows:
            residual = result["residuals"][control["control_id"]]
            if residual["status"].startswith("disabled"):
                continue
            connection.execute(
                """INSERT INTO scheme_controls
                (scheme_id,control_id,role,weight,residual_m,vector_json,locked)
                VALUES (?,?,?,?,?,?,?)""",
                (scheme_id, control["control_id"], residual["status"], residual["weight"], residual["norm"],
                 dumps(list(residual["vector"])), 1 if residual["locked"] else 0),
            )
        for corr in corr_rows:
            residual = result["residuals"][corr["correspondence_id"]]
            if residual["status"].startswith("disabled"):
                continue
            connection.execute(
                """INSERT INTO scheme_correspondences
                (scheme_id,correspondence_id,role,weight,residual_m,vector_json,group_id)
                VALUES (?,?,?,?,?,?,?)""",
                (scheme_id, corr["correspondence_id"], residual["status"], residual["weight"], residual["norm"],
                 dumps(list(residual["vector"])), corr["group_id"]),
            )
        _finish_job(connection, job_id, "solve", request["dataset_id"], scheme_id, request, input_fp, rules_fp, "succeeded", {"scheme_id": scheme_id})
        _audit(connection, job_id, "scheme_created", dataset_id=request["dataset_id"], scheme_id=scheme_id, message=request["label"])
    return {"replayed": False, "job": get_existing_job(connection, job_id), "scheme_id": scheme_id}


def publish_scheme(connection: sqlite3.Connection, request: dict[str, Any]) -> dict[str, Any]:
    job_id = request["job_id"]
    input_fp = fingerprint(request)
    existing = get_existing_job(connection, job_id)
    if existing:
        return _replay(existing, "publish", input_fp)
    with immediate_transaction(connection):
        row = one(connection, "SELECT * FROM schemes WHERE scheme_id=?", (request["scheme_id"],))
        if row is None:
            raise ServiceError(404, "scheme_not_found", "scheme not found")
        scheme = dict(row)
        if scheme["status"] != "published":
            connection.execute("UPDATE schemes SET status='published', published_at=? WHERE scheme_id=?", (now_iso(), scheme["scheme_id"]))
        _finish_job(connection, job_id, "publish", scheme["dataset_id"], scheme["scheme_id"], request, input_fp, scheme["rules_fingerprint"], "succeeded", {"scheme_id": scheme["scheme_id"]})
        _audit(connection, job_id, "scheme_published", dataset_id=scheme["dataset_id"], scheme_id=scheme["scheme_id"], message=scheme["label"])
    return {"replayed": False, "job": get_existing_job(connection, job_id), "scheme_id": scheme["scheme_id"]}


def _json_ready(value):
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    return value


def get_dataset_detail(connection, dataset_id: str) -> dict[str, Any]:
    dataset, strips, points, controls, correspondences = _load_dataset(connection, dataset_id)
    return {
        "dataset": {key: loads(value, value) if key.endswith("_json") else value for key, value in dataset.items()},
        "strips": [{key: loads(value, value) if key.endswith("_json") else value for key, value in strip.items()} for strip in strips],
        "points": points,
        "controls": controls,
        "correspondences": correspondences,
    }


def get_scheme_detail(connection, scheme_id: str) -> dict[str, Any]:
    row = one(connection, "SELECT * FROM schemes WHERE scheme_id=?", (scheme_id,))
    if row is None:
        raise ServiceError(404, "scheme_not_found", "scheme not found")
    scheme = dict(row)
    config = loads(scheme.pop("config_json"))
    result = loads(scheme.pop("result_json"))
    evidence_controls = many(connection, "SELECT * FROM scheme_controls WHERE scheme_id=? ORDER BY control_id", (scheme_id,))
    evidence_corr = many(connection, "SELECT * FROM scheme_correspondences WHERE scheme_id=? ORDER BY correspondence_id", (scheme_id,))
    dataset, strips, point_rows, _, _ = _load_dataset(connection, scheme["dataset_id"])
    transformed_points = []
    for point in point_rows:
        record = PointRecord(point["point_id"], point["strip_id"], point["time_value"], (point["x"], point["y"], point["z"]))
        seg_specs = result["segments"]
        from .adjustment import parameter_layout, segment_for_point, time_fraction
        from .geometry import apply_local_to_common, exp_so3
        layout = parameter_layout(seg_specs, result["model_version"])
        seg_index, segment = segment_for_point(record, seg_specs)
        segment_key = ("__common__", 0) if result["model_version"] == "rigid_v1" else (record.strip_id, seg_index)
        state = result["transform"]["segments"][f"{segment_key[0]}#segment{segment_key[1]}"]
        fraction = time_fraction(record, segment, seg_index) if result["model_version"] == "drift_v1" else 0.0
        slope = tuple(state["a"] or (0.0, 0.0, 0.0))
        drift = tuple(value * fraction for value in slope)
        predicted = apply_local_to_common(
            record.local,
            tuple(tuple(row) for row in result["transform"]["rotation"]),
            tuple(state["t"]),
            drift,
        )
        transformed_points.append({
            "point_id": point["point_id"], "strip_id": point["strip_id"], "time_value": point["time_value"],
            "segment_index": seg_index,
            "source_local": [point["x"], point["y"], point["z"]], "common": _json_ready(predicted),
        })
    return {
        "scheme": {**scheme, "config": config, "result": _json_ready(result)},
        "evidence_controls": [dict(row) | {"vector": loads(row["vector_json"])} for row in evidence_controls],
        "evidence_correspondences": [dict(row) | {"vector": loads(row["vector_json"])} for row in evidence_corr],
        "transformed_points": transformed_points,
    }


def compare_schemes(connection, left_id: str, right_id: str) -> dict[str, Any]:
    left = get_scheme_detail(connection, left_id)
    right = get_scheme_detail(connection, right_id)
    if left["scheme"]["dataset_id"] != right["scheme"]["dataset_id"]:
        raise ServiceError(422, "different_datasets", "only schemes for the same dataset can be compared")
    return {"left": left, "right": right}


def list_schemes(connection, dataset_id: str) -> list[dict[str, Any]]:
    rows = many(connection, "SELECT scheme_id,dataset_id,parent_scheme_id,label,model_version,status,rules_fingerprint,created_at,published_at FROM schemes WHERE dataset_id=? ORDER BY created_at", (dataset_id,))
    return [dict(row) for row in rows]


def list_datasets(connection) -> list[dict[str, Any]]:
    rows = many(connection, "SELECT dataset_id,name,source_name,common_crs,common_length_unit,input_fingerprint,created_at FROM datasets ORDER BY created_at")
    return [dict(row) for row in rows]


def get_job_detail(connection, job_id: str) -> dict[str, Any]:
    job = get_existing_job(connection, job_id)
    if job is None:
        raise ServiceError(404, "job_not_found", "job not found")
    events = many(connection, "SELECT audit_id,job_id,event_type,dataset_id,scheme_id,message,occurred_at FROM audit_events WHERE job_id=? ORDER BY audit_id", (job_id,))
    return {"job": job, "audit_events": [dict(row) for row in events]}

"""Service layer: import, plan derivation, deterministic job execution."""

from __future__ import annotations

import sqlite3
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from .fingerprint import RULE_VERSION
from .geo.models import (
    RIGID6,
    STRIP_DRIFT,
    STRIP_OFFSET3,
    build_layout,
)
from .geo.solver import (
    DEFAULT_CORR_WEIGHT,
    DEFAULT_GCP_WEIGHT,
    DEFAULT_LOCK_FACTOR,
    DEFAULT_OUTLIER_SIGMA,
    DEFAULT_SMOOTH_WEIGHT,
    DEFAULT_DRIFT_GAUGE_WEIGHT,
    CorrObs,
    GcpObs,
    solve,
)
from .db import (
    dumps,
    emit_audit,
    loads,
    transaction,
    utc_now,
    point_input_digest,
)

UNIT_TO_M = {
    "m": 1.0,
    "metre": 1.0,
    "meter": 1.0,
    "cm": 0.01,
    "mm": 0.001,
    "ft": 0.3048,
    "foot": 0.3048,
    "us_survey_ft": 1200.0 / 3937.0,
    "survey_ft": 1200.0 / 3937.0,
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "model_version": RIGID6,
    "drift_breaks": {},
    "weights": {"gcp": DEFAULT_GCP_WEIGHT, "corr": DEFAULT_CORR_WEIGHT,
                "smooth": DEFAULT_SMOOTH_WEIGHT, "drift_gauge": DEFAULT_DRIFT_GAUGE_WEIGHT},
    "lock_factor": DEFAULT_LOCK_FACTOR,
    "outlier_sigma": DEFAULT_OUTLIER_SIGMA,
    "reject_outliers": True,
    "disabled_gcp_ids": [],
    "locked_gcp_ids": [],
    "disabled_corr_ids": [],
    "disabled_corr_groups": [],
}


class ServiceError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def normalize_unit(unit: str) -> float:
    key = (unit or "").strip().lower()
    if key not in UNIT_TO_M:
        raise ServiceError(
            400,
            f"unknown unit {unit!r}; supported: {sorted(set(UNIT_TO_M))}",
        )
    return UNIT_TO_M[key]


def import_pointset(conn: sqlite3.Connection, payload: Dict[str, Any]) -> Dict[str, Any]:
    name = str(payload.get("name") or "pointset")  # noqa: F541 safeguard
    unit = str(payload.get("unit", "m"))
    scale = normalize_unit(unit)
    crs = payload.get("crs")
    source_description = payload.get("source_description")
    digest = point_input_digest(payload)

    existing = conn.execute(
        "SELECT id FROM pointsets WHERE input_digest=?", (digest,)
    ).fetchone()
    if existing:
        return {"id": existing["id"], "replayed": True}

    simulate_crash = bool(payload.get("simulate_crash"))
    with transaction(conn):
        cur = conn.execute(
            "INSERT INTO pointsets(name, source_unit, source_crs, "
            "source_description, unit_scale_to_m, created_at, input_digest) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (name, unit, crs, source_description, scale, utc_now(), digest),
        )
        pointset_id = cur.lastrowid

        seen_point_ids: set[int] = set()
        strip_uids: List[str] = []
        for strip in payload.get("strips", []):
            uid = str(strip["strip_id"])
            strip_uids.append(uid)
            scur = conn.execute(
                "INSERT INTO strips(pointset_id, strip_uid) VALUES (?, ?)",
                (pointset_id, uid),
            )
            strip_pk = scur.lastrowid
            for ordinal, point in enumerate(strip.get("points", [])):
                pid = int(point["id"])
                if pid in seen_point_ids:
                    raise ServiceError(400, f"duplicate point id {pid}")
                seen_point_ids.add(pid)
                conn.execute(
                    "INSERT INTO points(id, pointset_id, strip_pk, strip_uid, "
                    "ordinal, timestamp_s, local_x, local_y, local_z) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (pid, pointset_id, strip_pk, uid, ordinal,
                     point.get("timestamp_s"),
                     float(point["x"]), float(point["y"]), float(point["z"])),
                )
            if simulate_crash:
                raise RuntimeError("simulated crash during import")

        for gcp in payload.get("gcps", []):
            control_unit = str(gcp.get("control_unit", "m"))
            normalize_unit(control_unit)
            gcp_pid = int(gcp["point_id"])
            if gcp_pid not in seen_point_ids:
                raise ServiceError(
                    400, f"gcp {gcp.get('code')!r} references unknown point {gcp_pid}"
                )
            conn.execute(
                "INSERT INTO gcps(pointset_id, point_id, code, control_x, "
                "control_y, control_z, control_unit, control_crs) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (pointset_id, gcp_pid, str(gcp["code"]),
                 float(gcp["control_x"]), float(gcp["control_y"]),
                 float(gcp["control_z"]), control_unit, gcp.get("control_crs")),
            )
        for corr in payload.get("correspondences", []):
            a, b = int(corr["point_a_id"]), int(corr["point_b_id"])
            if a not in seen_point_ids or b not in seen_point_ids:
                raise ServiceError(400, "correspondence references unknown point")
            conn.execute(
                "INSERT INTO correspondences(pointset_id, point_a_id, point_b_id) "
                "VALUES (?, ?, ?)",
                (pointset_id, a, b),
            )
    return {"id": pointset_id, "replayed": False, "input_digest": digest,
            "strip_ids": strip_uids}


# --- read helpers -----------------------------------------------------------

def get_pointset(conn: sqlite3.Connection, pointset_id: int) -> Dict[str, Any]:
    row = conn.execute("SELECT * FROM pointsets WHERE id=?", (pointset_id,)).fetchone()
    if not row:
        raise ServiceError(404, f"pointset {pointset_id} not found")
    return dict(row)


def list_strip_points(conn: sqlite3.Connection, pointset_id: int) -> Dict[int, Dict[str, Any]]:
    """Return strip points in metres with deterministic along-track s."""
    scale = get_pointset(conn, pointset_id)["unit_scale_to_m"]
    strips: Dict[int, Dict[str, Any]] = {}
    rows = conn.execute(
        "SELECT p.id, p.strip_uid, p.ordinal, p.timestamp_s, p.local_x, "
        "p.local_y, p.local_z, s.id AS strip_pk FROM points p JOIN strips s "
        "ON s.id = p.strip_pk WHERE p.pointset_id=? ORDER BY p.strip_pk, p.ordinal",
        (pointset_id,),
    ).fetchall()
    grouped: Dict[int, List[sqlite3.Row]] = {}
    for row in rows:
        grouped.setdefault(row["strip_pk"], []).append(row)
    for strip_pk, points in grouped.items():
        have_time = all(p["timestamp_s"] is not None for p in points)
        xyz = [(p["local_x"] * scale, p["local_y"] * scale, p["local_z"] * scale)
               for p in points]
        if have_time:
            t0, t1 = points[0]["timestamp_s"], points[-1]["timestamp_s"]
            s_values = [
                0.0 if t1 == t0 else (p["timestamp_s"] - t0) / (t1 - t0)
                for p in points
            ]
        else:
            cumulative = [0.0]
            for i in range(1, len(xyz)):
                d = (
                    (xyz[i][0] - xyz[i - 1][0]) ** 2
                    + (xyz[i][1] - xyz[i - 1][1]) ** 2
                ) ** 0.5
                cumulative.append(cumulative[-1] + d)
            total = cumulative[-1]
            s_values = [0.0 if total == 0 else c / total for c in cumulative]
        strips[strip_pk] = {
            "strip_uid": points[0]["strip_uid"],
            "points": [
                {"id": p["id"], "s": float(s),
                 "xyz": [xyz[i][0], xyz[i][1], xyz[i][2]]}
                for i, (p, s) in enumerate(zip(points, s_values))
            ],
        }
    return strips


def strip_index_map(conn: sqlite3.Connection, pointset_id: int) -> Dict[str, int]:
    rows = conn.execute(
        "SELECT id, strip_uid FROM strips WHERE pointset_id=? ORDER BY id",
        (pointset_id,),
    ).fetchall()
    return {row["strip_uid"]: idx for idx, row in enumerate(rows)}


def build_observations(
    conn: sqlite3.Connection, pointset_id: int, config: Dict[str, Any]
) -> Tuple[List[str], List[GcpObs], List[CorrObs], List[Dict[str, Any]]]:
    get_pointset(conn, pointset_id)
    strip_order = [
        row["strip_uid"]
        for row in conn.execute(
            "SELECT strip_uid FROM strips WHERE pointset_id=? ORDER BY id",
            (pointset_id,),
        ).fetchall()
    ]
    strip_pos: Dict[str, int] = {uid: i for i, uid in enumerate(strip_order)}
    strips = list_strip_points(conn, pointset_id)
    point_lookup: Dict[int, Tuple[int, float, List[float]]] = {}
    for strip_pk, data in strips.items():
        idx = strip_pos[data["strip_uid"]]
        for point in data["points"]:
            point_lookup[point["id"]] = (idx, point["s"], point["xyz"])

    disabled_gcps = set(config.get("disabled_gcp_ids", []))
    locked_gcps = set(config.get("locked_gcp_ids", []))
    gcp_weight = float(config.get("weights", {}).get("gcp", DEFAULT_GCP_WEIGHT))
    corr_weight = float(config.get("weights", {}).get("corr", DEFAULT_CORR_WEIGHT))

    gcp_rows = conn.execute(
        "SELECT * FROM gcps WHERE pointset_id=? ORDER BY id", (pointset_id,)
    ).fetchall()
    gcps: List[GcpObs] = []
    gcp_details: List[Dict[str, Any]] = []
    for row in gcp_rows:
        if row["id"] in disabled_gcps and row["id"] not in locked_gcps:
            gcp_details.append({"id": row["id"], "code": row["code"],
                                "active": False, "locked": False})
            continue
        locked = row["id"] in locked_gcps
        strip_idx, s_val, local_xyz = point_lookup.get(int(row["point_id"]), (None, None, None))
        if local_xyz is None:
            gcp_details.append({"id": row["id"], "code": row["code"],
                                "active": False, "locked": locked,
                                "note": "no matching imported point"})
            continue
        control_scale = normalize_unit(row["control_unit"])
        control = [row["control_x"] * control_scale,
                   row["control_y"] * control_scale,
                   row["control_z"] * control_scale]
        gcps.append(GcpObs(
            point_id=int(row["id"]), strip_index=strip_idx, s=s_val,
            local_xyz=np.array(local_xyz), control_xyz=np.array(control),
            weight=gcp_weight, locked=locked,
        ))
        gcp_details.append({"id": row["id"], "code": row["code"],
                            "active": True, "locked": locked})

    disabled_corrs = set(config.get("disabled_corr_ids", []))
    disabled_groups = set(config.get("disabled_corr_groups", []))
    corr_rows = conn.execute(
        "SELECT * FROM correspondences WHERE pointset_id=? ORDER BY id",
        (pointset_id,),
    ).fetchall()
    corrs: List[CorrObs] = []
    corr_details: List[Dict[str, Any]] = []
    for row in corr_rows:
        a = point_lookup.get(row["point_a_id"])
        b = point_lookup.get(row["point_b_id"])
        group = f"{a[0]}:{b[0]}" if a and b else "?"
        entry = {"id": row["id"], "group": group,
                 "active": row["id"] not in disabled_corrs and group not in disabled_groups}
        corr_details.append(entry)
        if not entry["active"] or a is None or b is None:
            continue
        corrs.append(CorrObs(
            corr_id=int(row["id"]),
            strip_a=a[0], s_a=a[1], local_a=np.array(a[2]),
            strip_b=b[0], s_b=b[1], local_b=np.array(b[2]),
            weight=corr_weight,
        ))
    return strip_order, gcps, corrs, gcp_details


# --- plans ------------------------------------------------------------------

def get_plan(conn: sqlite3.Connection, plan_id: int) -> Dict[str, Any]:
    row = conn.execute("SELECT * FROM plans WHERE id=?", (plan_id,)).fetchone()
    if not row:
        raise ServiceError(404, f"plan {plan_id} not found")
    data = dict(row)
    data["config"] = loads(data.pop("config_json"))
    return data


def list_plans(conn: sqlite3.Connection, pointset_id: int) -> List[Dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM plans WHERE pointset_id=? ORDER BY id", (pointset_id,)
    ).fetchall()
    out = []
    for row in rows:
        data = dict(row)
        data["config"] = loads(data.pop("config_json"))
        out.append(data)
    return out


def canonicalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(DEFAULT_CONFIG)
    for key, value in config.items():
        if value is None:
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **{k: v for k, v in value.items()
                                            if v is not None}}
        else:
            merged[key] = value
    merged["model_version"] = str(merged["model_version"])
    merged["drift_breaks"] = {
        str(k): [float(x) for x in v]
        for k, v in sorted((merged.get("drift_breaks") or {}).items())
    }
    for list_key in ("disabled_gcp_ids", "locked_gcp_ids",
                     "disabled_corr_ids"):
        merged[list_key] = sorted(int(x) for x in merged.get(list_key, []))
    merged["disabled_corr_groups"] = sorted(
        str(x) for x in merged.get("disabled_corr_groups", [])
    )
    merged["weights"] = {k: float(v) for k, v in sorted(merged["weights"].items())}
    return merged


def create_plan(
    conn: sqlite3.Connection,
    pointset_id: int,
    name: str,
    config: Dict[str, Any],
    parent_plan_id: Optional[int] = None,
) -> int:
    get_pointset(conn, pointset_id)
    if parent_plan_id is not None:
        parent = get_plan(conn, parent_plan_id)
        if parent["pointset_id"] != pointset_id:
            raise ServiceError(400, "parent plan belongs to another pointset")
    config = canonicalize_config(config)
    if config["model_version"] not in (RIGID6, STRIP_OFFSET3, STRIP_DRIFT):
        raise ServiceError(400, f"unknown model version {config['model_version']}")
    if config["model_version"] == STRIP_DRIFT:
        from .geo.models import normalize_breaks
        for sid, breaks in config["drift_breaks"].items():
            normalize_breaks(breaks)
    cur = conn.execute(
        "INSERT INTO plans(pointset_id, name, model_version, config_json, "
        "parent_plan_id, published, rule_version, created_at) "
        "VALUES (?, ?, ?, ?, ?, 0, ?, ?)",
        (pointset_id, name, config["model_version"], dumps(config),
         parent_plan_id, RULE_VERSION, utc_now()),
    )
    plan_id = int(cur.lastrowid)
    emit_audit(
        conn,
        event_key=f"plan-created:{pointset_id}:{plan_id}",
        event_type="plan_created",
        payload={"plan_id": plan_id, "parent_plan_id": parent_plan_id,
                 "model_version": config["model_version"]},
        plan_id=plan_id,
    )
    return plan_id


def fork_plan(
    conn: sqlite3.Connection, plan_id: int, name: str, changes: Dict[str, Any]
) -> int:
    parent = get_plan(conn, plan_id)
    config = dict(parent["config"])
    for key, value in changes.items():
        config[key] = value
    return create_plan(
        conn, parent["pointset_id"], name, config, parent_plan_id=plan_id
    )


def publish_plan(conn: sqlite3.Connection, plan_id: int) -> None:
    plan = get_plan(conn, plan_id)
    if plan["published"]:
        return
    with transaction(conn):
        conn.execute("UPDATE plans SET published=1 WHERE id=?", (plan_id,))
        emit_audit(
            conn,
            event_key=f"plan-published:{plan_id}",
            event_type="plan_published",
            payload={"plan_id": plan_id},
            plan_id=plan_id,
        )


# --- jobs -------------------------------------------------------------------

def _pointset_solve_digest(conn: sqlite3.Connection, pointset_id: int) -> str:
    from .fingerprint import fingerprint
    rows = {
        "pointset": dict(get_pointset(conn, pointset_id)),
        "points": [
            dict(r) for r in conn.execute(
                "SELECT strip_uid, ordinal, timestamp_s, local_x, local_y, "
                "local_z FROM points WHERE pointset_id=? ORDER BY id",
                (pointset_id,),
            ).fetchall()
        ],
        "gcps": [dict(r) for r in conn.execute(
            "SELECT point_id, code, control_x, control_y, control_z, "
            "control_unit, control_crs FROM gcps WHERE pointset_id=? ORDER BY id",
            (pointset_id,),
        ).fetchall()],
        "correspondences": [dict(r) for r in conn.execute(
            "SELECT point_a_id, point_b_id FROM correspondences "
            "WHERE pointset_id=? ORDER BY id", (pointset_id,)
        ).fetchall()],
    }
    return fingerprint(rows)


def get_job(conn: sqlite3.Connection, job_id: int) -> Dict[str, Any]:
    row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if not row:
        raise ServiceError(404, f"job {job_id} not found")
    data = dict(row)
    data["result"] = loads(data.pop("result_json"))
    return data


def latest_result(conn: sqlite3.Connection, plan_id: int) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        "SELECT id, status, result_json FROM jobs WHERE plan_id=? AND status='succeeded' "
        "ORDER BY id DESC LIMIT 1", (plan_id,)
    ).fetchone()
    if not row:
        return None
    return {"job_id": row["id"], "result": loads(row["result_json"])}


def _execute_solve(
    conn: sqlite3.Connection, plan: Dict[str, Any]
) -> Dict[str, Any]:
    pointset_id = plan["pointset_id"]
    config = canonicalize_config(plan["config"])
    strip_order, gcps, corrs, gcp_details = build_observations(
        conn, pointset_id, config
    )
    layout = build_layout(
        config["model_version"], strip_order, config.get("drift_breaks")
    )
    result = solve(
        layout,
        gcps,
        corrs,
        smooth_weight=float(config["weights"]["smooth"]),
        gauge_weight=float(config["weights"].get(
            "drift_gauge", DEFAULT_DRIFT_GAUGE_WEIGHT)),
        outlier_sigma=float(config["outlier_sigma"]),
        reject_outliers=bool(config["reject_outliers"]),
    )
    return {
        "rule_version": RULE_VERSION,
        "model_version": config["model_version"],
        "status": result.status,
        "message": result.message,
        "converged": result.converged,
        "iterations": result.iterations,
        "rank": result.rank,
        "n_params": result.n_params,
        "singular_values": result.singular_values,
        "params": result.params,
        "parameter_labels": __import__(
            "app.geo.solver", fromlist=["parameter_labels"]
        ).parameter_labels(layout),
        "null_directions": result.null_directions,
        "null_labels": result.null_labels,
        "candidates": result.candidates,
        "rms_m": result.rms_m,
        "gcp_residuals": result.gcp_residuals,
        "corr_residuals": result.corr_residuals,
        "rejected_corr_ids": result.rejected_corr_ids,
        "gcp_participation": gcp_details,
        "config": config,
    }


def run_job(
    conn: sqlite3.Connection,
    plan_id: int,
    simulate_crash: bool = False,
) -> Tuple[Dict[str, Any], bool]:
    """Run (or replay) the solve job for a plan.

    Returns (job, replayed).  Replaying an identical job returns the stored
    row and emits no new audit event.

    Crash semantics use two durable checkpoints: the job is committed as
    ``queued`` (with its audit event) before solving, flipped to ``running``
    in its own commit, then marked ``succeeded``/``failed`` in a final
    commit.  An abnormal exit therefore never leaves a partially *changed
    point set* (points are immutable and untouched), and a stranded
    ``queued``/``running`` job is recovered as ``failed`` at startup.  The
    original inputs remain intact; the user fixes the cause and forks a new
    plan rather than mutating state in place.
    """
    plan = get_plan(conn, plan_id)
    pointset_id = plan["pointset_id"]
    config = canonicalize_config(plan["config"])
    input_digest = _pointset_solve_digest(conn, pointset_id)
    from .fingerprint import job_fingerprint
    fp = job_fingerprint(input_digest, config)

    existing = conn.execute(
        "SELECT * FROM jobs WHERE fingerprint=?", (fp,)
    ).fetchone()
    if existing:
        data = dict(existing)
        data["result"] = loads(data.pop("result_json"))
        return data, True

    # Checkpoint 1: durable queue record + audit.
    with transaction(conn):
        now = utc_now()
        cur = conn.execute(
            "INSERT INTO jobs(plan_id, fingerprint, status, result_json, error, "
            "created_at, updated_at) VALUES (?, ?, 'queued', NULL, NULL, ?, ?)",
            (plan_id, fp, now, now),
        )
        job_id = int(cur.lastrowid)
        emit_audit(
            conn,
            event_key=f"job-queued:{fp}",
            event_type="job_queued",
            payload={"plan_id": plan_id, "job_id": job_id},
            job_id=job_id,
            plan_id=plan_id,
        )

    # Checkpoint 2: running.
    with transaction(conn):
        conn.execute(
            "UPDATE jobs SET status='running', updated_at=? WHERE id=? "
            "AND status='queued'",
            (utc_now(), job_id),
        )

    # Solving itself never writes point data; only the terminal checkpoint
    # touches persistent state.
    try:
        if simulate_crash:
            raise RuntimeError("simulated crash during solve")
        result = _execute_solve(conn, plan)
    except Exception as exc:  # noqa: BLE001 - recorded as job failure
        with transaction(conn):
            conn.execute(
                "UPDATE jobs SET status='failed', error=?, updated_at=? WHERE id=?",
                (str(exc), utc_now(), job_id),
            )
            emit_audit(
                conn,
                event_key=f"job-failed:{fp}",
                event_type="job_failed",
                payload={"plan_id": plan_id, "job_id": job_id, "error": str(exc)},
                job_id=job_id,
                plan_id=plan_id,
            )
        raise

    # Checkpoint 3: durable result + audit.
    with transaction(conn):
        conn.execute(
            "UPDATE jobs SET status='succeeded', result_json=?, "
            "updated_at=? WHERE id=? AND status='running'",
            (dumps(result), utc_now(), job_id),
        )
        emit_audit(
            conn,
            event_key=f"job-succeeded:{fp}",
            event_type="job_succeeded",
            payload={"plan_id": plan_id, "job_id": job_id,
                     "solve_status": result["status"], "rank": result["rank"],
                     "n_params": result["n_params"]},
            job_id=job_id,
            plan_id=plan_id,
        )

    row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    data = dict(row)
    data["result"] = loads(data.pop("result_json"))
    return data, False


# --- corrected views --------------------------------------------------------

def corrected_points(
    conn: sqlite3.Connection, plan_id: int
) -> Dict[str, Any]:
    plan = get_plan(conn, plan_id)
    latest = latest_result(conn, plan_id)
    if latest is None:
        raise ServiceError(409, f"plan {plan_id} has no successful job yet")
    result = latest["result"]
    params = np.asarray(result["params"], dtype=np.float64)
    strip_order_rows = [
        row["strip_uid"]
        for row in conn.execute(
            "SELECT strip_uid FROM strips WHERE pointset_id=? ORDER BY id",
            (plan["pointset_id"],),
        ).fetchall()
    ]
    layout = build_layout(
        plan["config"]["model_version"],
        strip_order_rows,
        plan["config"].get("drift_breaks"),
    )
    from .geo.models import corrected_point
    strips = list_strip_points(conn, plan["pointset_id"])
    pos = {uid: i for i, uid in enumerate(strip_order_rows)}
    out_strips: Dict[str, List[Dict[str, Any]]] = {}
    for data in strips.values():
        uid = data["strip_uid"]
        idx = pos[uid]
        points_out = []
        for point in data["points"]:
            corrected = corrected_point(
                layout, params, np.asarray(point["xyz"]), idx, point["s"]
            )
            points_out.append({
                "id": point["id"], "s": point["s"],
                "local_xyz_m": point["xyz"],
                "corrected_xyz_m": [float(v) for v in corrected],
            })
        out_strips[uid] = points_out
    return {"plan_id": plan_id, "unit": "m", "strips": out_strips}


def audit_log(conn: sqlite3.Connection, limit: int = 200) -> List[Dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM audit_events ORDER BY id DESC LIMIT ?", (int(limit),)
    ).fetchall()
    return [
        {"id": r["id"], "event_type": r["event_type"], "job_id": r["job_id"],
         "plan_id": r["plan_id"], "payload": loads(r["payload_json"]),
         "created_at": r["created_at"]}
        for r in rows
    ]

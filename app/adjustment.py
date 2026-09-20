"""Deterministic point-cloud strip adjustment models."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from .geometry import IDENTITY, ZERO, Matrix3, Vector3, apply_local_to_common, exp_so3, matmul, matvec, segment_index
from .linalg import constrained_least_squares, rank_revealing_least_squares

ModelName = Literal["rigid_v1", "strip_offset_v1", "drift_v1"]

MODEL_LABELS = {
    "rigid_v1": "Common rotation and common translation",
    "strip_offset_v1": "Common rotation with one translation per segment",
    "drift_v1": "Common rotation with translation and linear time drift per segment",
}


@dataclass(frozen=True)
class PointRecord:
    point_id: str
    strip_id: str
    time_value: float
    local: Vector3


@dataclass(frozen=True)
class ControlRecord:
    control_id: str
    point_id: str
    common: Vector3
    weight: float = 1.0


@dataclass(frozen=True)
class CorrespondenceRecord:
    correspondence_id: str
    left_point_id: str
    right_point_id: str
    weight: float = 1.0
    group_id: str | None = None


@dataclass(frozen=True)
class SegmentSpec:
    strip_id: str
    split_time: float | None = None


def _validated_point(point_id: str, points: dict[str, PointRecord]) -> PointRecord:
    try:
        return points[point_id]
    except KeyError as exc:
        raise ValueError(f"unknown point {point_id}") from exc


def canonical_segments(strips: list[dict[str, Any]], overrides: dict[str, float | None]) -> list[dict[str, Any]]:
    result = []
    for strip in sorted(strips, key=lambda row: row["strip_id"]):
        strip_id = strip["strip_id"]
        start = float(strip["time_start"])
        end = float(strip["time_end"])
        if end <= start:
            raise ValueError(f"strip {strip_id} must have time_end > time_start")
        split = overrides.get(strip_id, strip.get("split_time"))
        boundary = None if split is None else float(split)
        if boundary is not None and not (start < boundary < end):
            raise ValueError(f"split for {strip_id} must lie strictly between time_start and time_end")
        result.append({"strip_id": strip_id, "time_start": start, "time_end": end, "split_time": boundary})
    return result


def segment_for_point(point: PointRecord, segments: list[dict[str, Any]]) -> tuple[int, dict[str, Any]]:
    matches = [segment for segment in segments if segment["strip_id"] == point.strip_id]
    if not matches:
        raise ValueError(f"point {point.point_id} references unknown strip {point.strip_id}")
    segment = matches[0]
    boundaries = [] if segment["split_time"] is None else [segment["split_time"]]
    index = segment_index(point.time_value, boundaries)
    if point.time_value < segment["time_start"] or point.time_value > segment["time_end"]:
        raise ValueError(f"point {point.point_id} is outside strip {point.strip_id} time range")
    return index, segment


def segment_keys(segments: list[dict[str, Any]], model: ModelName) -> list[tuple[str, int]]:
    if model == "rigid_v1":
        return [("__common__", 0)]
    keys: list[tuple[str, int]] = []
    for segment in sorted(segments, key=lambda row: row["strip_id"]):
        count = 1 if segment["split_time"] is None else 2
        keys.extend((segment["strip_id"], index) for index in range(count))
    return keys


def time_fraction(point: PointRecord, segment: dict[str, Any], index: int) -> float:
    split = segment["split_time"]
    if split is None:
        start, end = segment["time_start"], segment["time_end"]
    elif index == 0:
        start, end = segment["time_start"], split
    else:
        start, end = split, segment["time_end"]
    return 0.0 if end == start else (point.time_value - start) / (end - start)


def parameter_layout(segments: list[dict[str, Any]], model: ModelName) -> dict[str, Any]:
    keys = segment_keys(segments, model)
    layout = {"omega": (0, 3), "segments": {}}
    offset = 3
    if model == "rigid_v1":
        layout["segments"][keys[0]] = {"t": (offset, offset + 3), "a": None}
        offset += 3
    else:
        for key in keys:
            layout["segments"][key] = {"t": (offset, offset + 3), "a": None}
            offset += 3
            if model == "drift_v1":
                layout["segments"][key]["a"] = (offset, offset + 3)
                offset += 3
    layout["length"] = offset
    layout["segment_keys"] = keys
    return layout


def log_so3(rotation: Matrix3) -> Vector3:
    angle = math.acos(max(-1.0, min(1.0, (sum(rotation[i][i] for i in range(3)) - 1.0) / 2.0)))
    if abs(angle) < 1e-12:
        return (
            (rotation[2][1] - rotation[1][2]) / 2.0,
            (rotation[0][2] - rotation[2][0]) / 2.0,
            (rotation[1][0] - rotation[0][1]) / 2.0,
        )
    factor = angle / (2.0 * math.sin(angle))
    return (
        factor * (rotation[2][1] - rotation[1][2]),
        factor * (rotation[0][2] - rotation[2][0]),
        factor * (rotation[1][0] - rotation[0][1]),
    )


def _weight(weight: Any) -> float:
    value = float(weight)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("weights must be finite positive numbers")
    return value


def _options(config: dict[str, Any]) -> dict[str, Any]:
    disabled_controls = set(config.get("disabled_control_ids") or [])
    locked_controls = set(config.get("locked_control_ids") or [])
    disabled_correspondences = set(config.get("disabled_correspondence_ids") or [])
    disabled_groups = set(config.get("disabled_correspondence_group_ids") or [])
    overlap = disabled_controls & locked_controls
    if overlap:
        raise ValueError(f"control cannot be disabled and locked: {sorted(overlap)[0]}")
    limit = config.get("residual_limit_m")
    if limit is not None:
        limit = float(limit)
        if not math.isfinite(limit) or limit <= 0.0:
            raise ValueError("residual_limit_m must be a positive finite number")
    return {
        "disabled_controls": disabled_controls,
        "locked_controls": locked_controls,
        "disabled_correspondences": disabled_correspondences,
        "disabled_groups": disabled_groups,
        "weights": dict(config.get("weights") or {}),
        "limit": limit,
        "max_iterations": int(config.get("max_iterations", 20)),
        "tolerance": float(config.get("step_tolerance", 1e-11)),
    }


def _state_prediction(
    rotation: Matrix3,
    states: dict[tuple[str, int], dict[str, Vector3]],
    point: PointRecord,
    segments: list[dict[str, Any]],
    model: ModelName,
) -> tuple[Vector3, tuple[str, int], float]:
    seg_index, segment = segment_for_point(point, segments)
    key = ("__common__", 0) if model == "rigid_v1" else (point.strip_id, seg_index)
    fraction = time_fraction(point, segment, seg_index) if model == "drift_v1" else 0.0
    state = states[key]
    drift = tuple(value * fraction for value in state["a"])  # type: ignore[arg-type]
    predicted = apply_local_to_common(point.local, rotation, state["t"], drift)
    return predicted, key, fraction


def _jacobian_at(
    rotation: Matrix3,
    drift_vector: Vector3,
    point: PointRecord,
    key: tuple[str, int],
    fraction: float,
    model: ModelName,
    layout: dict[str, Any],
) -> list[list[float]]:
    shifted = (
        point.local[0] + drift_vector[0],
        point.local[1] + drift_vector[1],
        point.local[2] + drift_vector[2],
    )
    rotated_shifted = matvec(rotation, shifted)
    cross = (
        (0.0, rotated_shifted[2], -rotated_shifted[1]),
        (-rotated_shifted[2], 0.0, rotated_shifted[0]),
        (rotated_shifted[1], -rotated_shifted[0], 0.0),
    )
    jacobian = [[0.0] * layout["length"] for _ in range(3)]
    for axis in range(3):
        jacobian[axis][0:3] = cross[axis]
        t_start, _ = layout["segments"][key]["t"]
        jacobian[axis][t_start + axis] = 1.0
        if model == "drift_v1":
            a_start, _ = layout["segments"][key]["a"]
            for slope_axis in range(3):
                jacobian[axis][a_start + slope_axis] = rotation[axis][slope_axis] * fraction
    return jacobian


def _build_states(segments: list[dict[str, Any]], model: ModelName):
    keys = segment_keys(segments, model)
    states = {
        key: {"t": (0.0, 0.0, 0.0), "a": (0.0, 0.0, 0.0)}
        for key in keys
    }
    return states


def _state_vector(rotation: Matrix3, states: dict[tuple[str, int], dict[str, Vector3]], layout: dict[str, Any]) -> list[float]:
    vector = [0.0] * layout["length"]
    omega = log_so3(rotation)
    vector[0:3] = omega
    for key, entry in layout["segments"].items():
        start, _ = entry["t"]
        vector[start:start + 3] = states[key]["t"]
        if entry["a"] is not None:
            a_start, _ = entry["a"]
            vector[a_start:a_start + 3] = states[key]["a"]
    return vector


def _update_states(rotation: Matrix3, states, delta: list[float], segments, model, layout):
    delta_rotation = exp_so3((delta[0], delta[1], delta[2]))
    rotation = matmul(delta_rotation, rotation)
    for key, entry in layout["segments"].items():
        start, _ = entry["t"]
        current = states[key]["t"]
        states[key]["t"] = tuple(current[axis] + delta[start + axis] for axis in range(3))  # type: ignore[assignment]
        if entry["a"] is not None:
            a_start, _ = entry["a"]
            current_a = states[key]["a"]
            states[key]["a"] = tuple(current_a[axis] + delta[a_start + axis] for axis in range(3))  # type: ignore[assignment]
    return rotation


def _active_controls(controls, options):
    active = []
    for control in sorted(controls, key=lambda row: row.control_id):
        if control.control_id in options["disabled_controls"]:
            continue
        weight = options["weights"].get(f"control:{control.control_id}", control.weight)
        active.append((control, _weight(weight)))
    return active


def _active_correspondences(correspondences, points, options, rejected=None):
    rejected = rejected or set()
    active = []
    for correspondence in sorted(correspondences, key=lambda row: row.correspondence_id):
        if correspondence.correspondence_id in options["disabled_correspondences"]:
            continue
        if correspondence.group_id in options["disabled_groups"]:
            continue
        if correspondence.correspondence_id in rejected:
            continue
        weight = options["weights"].get(
            f"correspondence:{correspondence.correspondence_id}", correspondence.weight
        )
        left = _validated_point(correspondence.left_point_id, points)
        right = _validated_point(correspondence.right_point_id, points)
        active.append((correspondence, left, right, _weight(weight)))
    return active


def _build_iteration_system(
    rotation, states, points, controls, correspondences, segments, model, layout, options, rejected=None
):
    design: list[list[float]] = []
    response: list[float] = []
    rows: list[dict[str, Any]] = []
    equality: list[list[float]] = []
    equality_rhs: list[float] = []
    locked = set()

    for control, weight in _active_controls(controls, options):
        point = _validated_point(control.point_id, points)
        predicted, key, fraction = _state_prediction(rotation, states, point, segments, model)
        drift = tuple(value * fraction for value in states[key]["a"])
        jacobian = _jacobian_at(rotation, drift, point, key, fraction, model, layout)
        residual = tuple(control.common[axis] - predicted[axis] for axis in range(3))
        if control.control_id in options["locked_controls"]:
            equality.extend([row[:] for row in jacobian])
            equality_rhs.extend(residual)
            locked.add(control.control_id)
        else:
            for axis in range(3):
                design.append([math.sqrt(weight) * value for value in jacobian[axis]])
                response.append(math.sqrt(weight) * residual[axis])
                rows.append({"kind": "control", "id": control.control_id, "axis": axis, "weight": weight})

    for correspondence, left, right, weight in _active_correspondences(
        correspondences, points, options, rejected
    ):
        left_pred, left_key, left_fraction = _state_prediction(rotation, states, left, segments, model)
        right_pred, right_key, right_fraction = _state_prediction(rotation, states, right, segments, model)
        left_drift = tuple(value * left_fraction for value in states[left_key]["a"])
        right_drift = tuple(value * right_fraction for value in states[right_key]["a"])
        jl = _jacobian_at(rotation, left_drift, left, left_key, left_fraction, model, layout)
        jr = _jacobian_at(rotation, right_drift, right, right_key, right_fraction, model, layout)
        residual = tuple(right_pred[axis] - left_pred[axis] for axis in range(3))
        for axis in range(3):
            # RHS is right-left; using Jl-Jr keeps the consistent Gauss-Newton equation
            # Jl*d_l - Jr*d_r = right_pred-left_pred for matching right to left.
            row = [jl[axis][j] - jr[axis][j] for j in range(layout["length"])]
            design.append([math.sqrt(weight) * value for value in row])
            response.append(math.sqrt(weight) * residual[axis])
            rows.append({"kind": "correspondence", "id": correspondence.correspondence_id, "axis": axis, "weight": weight})

    return design, response, rows, (equality if equality else None), (equality_rhs if equality_rhs else None), locked


def _final_residuals(rotation, states, points, controls, correspondences, segments, model, options, rejected):
    residuals: dict[str, dict[str, Any]] = {}
    for control in sorted(controls, key=lambda row: row.control_id):
        if control.control_id in options["disabled_controls"]:
            residuals[control.control_id] = {
                "kind": "control", "status": "disabled", "vector": None, "norm": None, "weight": 0.0, "locked": False
            }
            continue
        point = _validated_point(control.point_id, points)
        predicted, _, _ = _state_prediction(rotation, states, point, segments, model)
        vector = tuple(control.common[axis] - predicted[axis] for axis in range(3))
        weight = options["weights"].get(f"control:{control.control_id}", control.weight)
        residuals[control.control_id] = {
            "kind": "control",
            "status": "locked" if control.control_id in options["locked_controls"] else "accepted",
            "vector": vector,
            "norm": math.sqrt(sum(value * value for value in vector)),
            "weight": _weight(weight),
            "locked": control.control_id in options["locked_controls"],
            "predicted": predicted,
        }

    for correspondence in sorted(correspondences, key=lambda row: row.correspondence_id):
        if correspondence.correspondence_id in options["disabled_correspondences"]:
            status = "disabled"
        elif correspondence.group_id in options["disabled_groups"]:
            status = "disabled_group"
        elif correspondence.correspondence_id in rejected:
            status = "rejected"
        else:
            status = "accepted"
        left = _validated_point(correspondence.left_point_id, points)
        right = _validated_point(correspondence.right_point_id, points)
        left_pred, _, _ = _state_prediction(rotation, states, left, segments, model)
        right_pred, _, _ = _state_prediction(rotation, states, right, segments, model)
        vector = tuple(right_pred[axis] - left_pred[axis] for axis in range(3))
        weight = options["weights"].get(
            f"correspondence:{correspondence.correspondence_id}", correspondence.weight
        )
        residuals[correspondence.correspondence_id] = {
            "kind": "correspondence",
            "status": status,
            "vector": vector,
            "norm": math.sqrt(sum(value * value for value in vector)),
            "weight": 0.0 if status.startswith("disabled") else _weight(weight),
            "group_id": correspondence.group_id,
            "left_predicted": left_pred,
            "right_predicted": right_pred,
        }
    return residuals


def _control_geometry(active_controls, points):
    coords = []
    for control, _ in active_controls:
        point = points[control.point_id]
        coords.append(point.local)
    if not coords:
        return 0, []
    centroid = tuple(sum(p[axis] for p in coords) / len(coords) for axis in range(3))
    centered = [tuple(p[axis] - centroid[axis] for axis in range(3)) for p in coords]
    qr_result = rank_revealing_least_squares(
        [list(point) for point in centered], [0.0] * len(centered), tolerance=1e-12
    )
    return qr_result["rank"], centered


def _parameter_names(layout, model):
    names = ["omega_x", "omega_y", "omega_z"]
    for key in sorted(layout["segments"], key=lambda value: (value[0], value[1])):
        strip_id, index = key
        label = "common" if model == "rigid_v1" else f"{strip_id}#segment{index}"
        names.extend([f"t_x:{label}", f"t_y:{label}", f"t_z:{label}"])
        if model == "drift_v1":
            names.extend([f"a_x:{label}", f"a_y:{label}", f"a_z:{label}"])
    return names


def solve_adjustment(
    model: ModelName,
    strips: list[dict[str, Any]],
    points: list[PointRecord] | dict[str, PointRecord],
    controls: list[ControlRecord],
    correspondences: list[CorrespondenceRecord],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if model not in MODEL_LABELS:
        raise ValueError(f"unknown model version {model}")
    config = config or {}
    options = _options(config)
    if model == "rigid_v1" and config.get("segment_splits"):
        raise ValueError("segment_splits are not applicable to rigid_v1")
    segments = canonical_segments(strips, dict(config.get("segment_splits") or {}))
    point_map = {point.point_id: point for point in points} if isinstance(points, list) else points
    layout = parameter_layout(segments, model)
    states = _build_states(segments, model)
    rotation = IDENTITY
    rejected: set[str] = set()
    iterations = []
    last_result = None

    for iteration in range(options["max_iterations"] + 1):
        design, response, row_meta, equality, equality_rhs, locked = _build_iteration_system(
            rotation, states, point_map, controls, correspondences, segments, model, layout, options, rejected
        )
        last_result = constrained_least_squares(design, response, equality, equality_rhs)
        if last_result.get("status") == "inconsistent_constraints":
            return {
                "status": "failed",
                "error_code": "inconsistent_locked_controls",
                "message": "locked controls cannot all be satisfied",
                "rank": last_result["rank"],
                "degrees_of_freedom": layout["length"],
                "unconstrained_directions": [],
                "candidates": [],
                "rejected_correspondence_ids": sorted(rejected),
            }
        delta = last_result["solution"]
        iterations.append({"rank": last_result["rank"], "delta_norm": math.sqrt(sum(v * v for v in delta))})
        if last_result["rank"] < layout["length"]:
            break
        if math.sqrt(sum(v * v for v in delta)) <= options["tolerance"]:
            break
        if iteration == options["max_iterations"]:
            return {
                "status": "failed",
                "error_code": "did_not_converge",
                "message": "Gauss-Newton iteration reached its limit",
                "iterations": iterations,
            }
        rotation = _update_states(rotation, states, delta, segments, model, layout)

    limit = options["limit"]
    pre_residuals = _final_residuals(
        rotation, states, point_map, controls, correspondences, segments, model, options, set()
    )
    if limit is not None:
        newly_rejected = {
            key
            for key, value in pre_residuals.items()
            if value["kind"] == "correspondence"
            and value["status"] == "accepted"
            and value["norm"] > limit
        }
        if newly_rejected:
            rejected |= newly_rejected
            states = _build_states(segments, model)
            rotation = IDENTITY
            iterations = []
            for iteration in range(options["max_iterations"] + 1):
                design, response, row_meta, equality, equality_rhs, locked = _build_iteration_system(
                    rotation, states, point_map, controls, correspondences, segments, model, layout, options, rejected
                )
                last_result = constrained_least_squares(design, response, equality, equality_rhs)
                if last_result.get("status") == "inconsistent_constraints":
                    return {
                        "status": "failed",
                        "error_code": "inconsistent_locked_controls_after_rejection",
                        "message": "rejected correspondences leave locked controls that cannot all be satisfied",
                        "rank": last_result["rank"],
                        "degrees_of_freedom": layout["length"],
                        "unconstrained_directions": [],
                        "candidates": [],
                        "rejected_correspondence_ids": sorted(rejected),
                    }
                delta = last_result["solution"]
                iterations.append({"rank": last_result["rank"], "delta_norm": math.sqrt(sum(v * v for v in delta))})
                if last_result["rank"] < layout["length"]:
                    break
                if math.sqrt(sum(v * v for v in delta)) <= options["tolerance"]:
                    break
                if iteration == options["max_iterations"]:
                    return {
                        "status": "failed",
                        "error_code": "did_not_converge_after_rejection",
                        "message": "Gauss-Newton iteration reached its limit after rejecting correspondences",
                        "iterations": iterations,
                        "rejected_correspondence_ids": sorted(rejected),
                    }
                rotation = _update_states(rotation, states, delta, segments, model, layout)

    names = _parameter_names(layout, model)
    unconstrained = []
    for vector in last_result["null_basis"]:
        active = [j for j, value in enumerate(vector) if abs(value) > 1e-12]
        representative = names[active[0]] if active else "unknown"
        unconstrained.append({
            "name": representative,
            "coefficients": {names[j]: vector[j] for j in active},
        })
    params = _state_vector(rotation, states, layout)
    candidates = [{"label": "free_parameters_zero", "parameters": params}]
    for index, basis in enumerate(last_result["null_basis"]):
        for scale, label in ((1.0, "nullspace+unit"), (-1.0, "nullspace-unit")):
            candidates.append({
                "label": f"{label}:{index}:{names[next(i for i, v in enumerate(basis) if abs(v) > 1e-12)]}",
                "parameters": [params[j] + scale * basis[j] for j in range(len(params))],
            })
    residuals = _final_residuals(
        rotation, states, point_map, controls, correspondences, segments, model, options, rejected
    )
    active_control_records = _active_controls(controls, options)
    geometry_rank, centered_controls = _control_geometry(active_control_records, point_map)
    warnings = []
    control_geometry_degenerate = bool(active_control_records and geometry_rank < 2)
    if control_geometry_degenerate:
        warnings.append("active control points are coincident or collinear and do not span a 3D pose")

    status = "ok" if last_result["rank"] == layout["length"] else "degenerate"
    return {
        "status": status,
        "error_code": None if status == "ok" else "rank_deficient",
        "model": model,
        "model_version": model,
        "model_description": MODEL_LABELS[model],
        "rank": last_result["rank"],
        "degrees_of_freedom": layout["length"],
        "unconstrained_directions": unconstrained,
        "candidates": candidates,
        "segments": segments,
        "transform": {
            "direction": "local_to_common",
            "equation": "common = R * (local + drift) + t",
            "omega": params[0:3],
            "rotation": exp_so3((params[0], params[1], params[2])),
            "segments": {
                f"{key[0]}#segment{key[1]}": {
                    "t": states[key]["t"],
                    "a": states[key]["a"] if model == "drift_v1" else None,
                }
                for key in layout["segment_keys"]
            },
        },
        "residuals": residuals,
        "iterations": iterations,
        "warnings": warnings,
        "control_geometry_rank": geometry_rank,
        "control_geometry_degenerate": control_geometry_degenerate,
        "locked_control_ids": sorted(options["locked_controls"]),
        "rejected_correspondence_ids": sorted(rejected),
        "residual_limit_m": limit,
        "parameter_names": names,
    }

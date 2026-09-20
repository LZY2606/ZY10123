"""Deterministic weighted least-squares alignment core.

The solver stacks three row groups (all residuals in **metres**):

* GCP rows:  ``w * (corrected_point - gcp_control)``
* correspondence rows: ``w * (corrected_a - corrected_b)``
* smoothness rows (drift model only): a regular *observation* row
  ``w_smooth * (d_{k+1} - d_k - rate_k * delta_b)`` that constrains the
  drift to be slowly changing.  It is a normal weighted observation with a
  documented weight (default 1.0), never hidden Tikhonov regularization.

Updates are Gauss-Newton.  For rigid6 the update is a *local* SE(3) twist
(so rotations stay exact regardless of magnitude); linear models solve in
one iteration.  Normal equations are inverted by SVD.  When the rank is
deficient the solver reports the rank, singular values, the unconstrained
(null-space) parameter directions and concrete feasible candidates -- it
never adds regularization to pick one silently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .models import (
    RIGID6,
    STRIP_DRIFT,
    STRIP_OFFSET3,
    ModelLayout,
    corrected_point,
)
from .transforms import RigidTransform

RANK_TOL = 1e-9
MAX_ITERATIONS = 20
STEP_TOL = 1e-10
DEFAULT_GCP_WEIGHT = 10.0
DEFAULT_CORR_WEIGHT = 1.0
DEFAULT_LOCK_FACTOR = 1e4
DEFAULT_SMOOTH_WEIGHT = 1.0
DEFAULT_DRIFT_GAUGE_WEIGHT = 100.0
DEFAULT_OUTLIER_SIGMA = 0.5
MAX_REJECT_PASSES = 10
CANDIDATE_STEP = 1.0


@dataclass
class GcpObs:
    point_id: int
    strip_index: int
    s: float
    local_xyz: np.ndarray
    control_xyz: np.ndarray
    weight: float = DEFAULT_GCP_WEIGHT
    locked: bool = False


@dataclass
class CorrObs:
    corr_id: int
    strip_a: int
    s_a: float
    local_a: np.ndarray
    strip_b: int
    s_b: float
    local_b: np.ndarray
    weight: float = DEFAULT_CORR_WEIGHT


@dataclass
class SolveResult:
    converged: bool
    iterations: int
    rank: int
    n_params: int
    singular_values: List[float]
    params: List[float]
    null_directions: List[List[float]]
    null_labels: List[str]
    candidates: List[dict]
    rms_m: float
    gcp_residuals: List[dict] = field(default_factory=list)
    corr_residuals: List[dict] = field(default_factory=list)
    rejected_corr_ids: List[int] = field(default_factory=list)
    status: str = "ok"
    message: str = ""


def parameter_labels(layout: ModelLayout) -> List[str]:
    if layout.model_version == RIGID6:
        return ["global.tx", "global.ty", "global.tz",
                "global.rx", "global.ry", "global.rz"]
    if layout.model_version == STRIP_OFFSET3:
        labels: List[str] = []
        for sid in layout.strip_ids:
            labels += [f"{sid}.tx", f"{sid}.ty", f"{sid}.tz"]
        return labels
    labels = ["global.tx", "global.ty", "global.tz",
              "global.rx", "global.ry", "global.rz"]
    for sid in layout.strip_ids:
        dlayout = layout.drift[sid]
        for k in range(dlayout.n_knots):
            labels += [f"{sid}.d{k}.x", f"{sid}.d{k}.y", f"{sid}.d{k}.z"]
    return labels


def rotation_jacobian(corrected_rotated: np.ndarray) -> np.ndarray:
    """Rotation columns of the local-twist Jacobian: -[q]_x with q the
    current corrected (translated) point.

    Verified by finite differences for the right-composed update
    T_new = T o Exp(delta): translation columns are the identity and the
    rotational columns equal -skew(q) evaluated at the current point.
    """
    x, y, z = (
        float(corrected_rotated[0]),
        float(corrected_rotated[1]),
        float(corrected_rotated[2]),
    )
    return np.array(
        [[0.0, z, -y], [-z, 0.0, x], [y, -x, 0.0]], dtype=np.float64
    )


def point_jacobian(
    layout: ModelLayout,
    params: np.ndarray,
    point: np.ndarray,
    strip_index: int,
    s: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (residual_linearization_point, Jacobian w.r.t. local params)."""
    n = layout.n_params
    jac = np.zeros((3, n), dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    version = layout.model_version

    if version == RIGID6:
        transform = RigidTransform.from_params(p[:6])
        corrected = transform.apply(point.reshape(1, 3))[0]
        # Local update T o Exp(dt, omega): translation columns are the
        # identity; rotation columns skew the rotated (untranslated) point.
        jac[:, 0:3] = np.eye(3)
        jac[:, 3:6] = rotation_jacobian(corrected)
        return corrected, jac

    if version == STRIP_OFFSET3:
        strip_index = int(strip_index)
        base = 3 * strip_index
        corrected = point + p[base:base + 3]
        jac[:, base:base + 3] = np.eye(3)
        return corrected, jac

    transform = RigidTransform.from_params(p[:6])
    sid = layout.strip_ids[strip_index]
    strip_index = int(strip_index)
    _, left, right, u = layout.drift[sid].segment_at(float(s))
    il = layout.drift_knot_index(sid, left)
    ir = layout.drift_knot_index(sid, right)
    rotated = transform.apply(point.reshape(1, 3))[0]
    drift = (1.0 - u) * p[il:il + 3] + u * p[ir:ir + 3]
    corrected = rotated + drift
    jac[:, 0:3] = np.eye(3)
    jac[:, 3:6] = rotation_jacobian(rotated)
    jac[:, il:il + 3] = (1.0 - u) * np.eye(3)
    if ir != il:
        jac[:, ir:ir + 3] = u * np.eye(3)
    else:
        # Endpoint points (u == 1 on a left segment) load exactly one knot.
        jac[:, ir:ir + 3] = 0.0
    return corrected, jac


def _smooth_rows(
    layout: ModelLayout, params: np.ndarray, smooth_weight: float,
    gauge_weight: float = DEFAULT_DRIFT_GAUGE_WEIGHT,
) -> Tuple[np.ndarray, np.ndarray]:
    """Drift model observations (all weighted, all user-visible).

    Two kinds of ordinary observation rows are produced:

    * *smoothness* rows ``w_smooth * (d_{k+1} - d_k)``: the "slowly varying
      drift" statement, default weight 1.0.
    * *gauge* rows ``w_gauge * d_0`` for the first knot of every strip: the
      piecewise drift and the global translation share a gauge freedom
      (adding a constant to all knot translations and subtracting it from
      the global translation leaves every point unchanged).  Observations
      alone cannot observe this mode, so an explicit datum choice is
      required; it appears here as a normal weighted observation with a
      documented default (100.0), never as hidden regularization.
    """
    rows_j: List[np.ndarray] = []
    rows_r: List[np.ndarray] = []
    if layout.model_version != STRIP_DRIFT:
        return np.zeros((0, layout.n_params)), np.zeros(0)
    for sid in layout.strip_ids:
        dlayout = layout.drift[sid]
        if gauge_weight > 0.0:
            i0 = layout.drift_knot_index(sid, 0)
            for axis in range(3):
                jac = np.zeros(layout.n_params, dtype=np.float64)
                jac[i0 + axis] = gauge_weight
                rows_j.append(jac)
                rows_r.append(gauge_weight * params[i0 + axis])
        if smooth_weight > 0.0:
            for k in range(dlayout.n_knots - 1):
                il = layout.drift_knot_index(sid, k)
                ir = layout.drift_knot_index(sid, k + 1)
                for axis in range(3):
                    jac = np.zeros(layout.n_params, dtype=np.float64)
                    jac[il + axis] = -smooth_weight
                    jac[ir + axis] = smooth_weight
                    residual = smooth_weight * (params[ir + axis] - params[il + axis])
                    rows_j.append(jac)
                    rows_r.append(residual)
    if not rows_j:
        return np.zeros((0, layout.n_params)), np.zeros(0)
    return np.vstack(rows_j), np.asarray(rows_r)


def _build_system(
    layout: ModelLayout,
    params: np.ndarray,
    gcps: Sequence[GcpObs],
    corrs: Sequence[CorrObs],
    smooth_weight: float,
    gauge_weight: float = DEFAULT_DRIFT_GAUGE_WEIGHT,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    n = layout.n_params
    jac_rows: List[np.ndarray] = []
    residual_rows: List[np.ndarray] = []
    meta = {"gcp_slices": {}, "corr_slices": {}}
    cursor = 0

    for gcp in gcps:
        corrected, jac = point_jacobian(
            layout, params, gcp.local_xyz, gcp.strip_index, gcp.s
        )
        weight = gcp.weight * (DEFAULT_LOCK_FACTOR if gcp.locked else 1.0)
        residual = weight * (corrected - gcp.control_xyz)
        jac_rows.append(weight * jac)
        residual_rows.append(residual)
        meta["gcp_slices"][gcp.point_id] = slice(cursor, cursor + 3)
        cursor += 3

    for corr in corrs:
        corrected_a, jac_a = point_jacobian(
            layout, params, corr.local_a, corr.strip_a, corr.s_a
        )
        corrected_b, jac_b = point_jacobian(
            layout, params, corr.local_b, corr.strip_b, corr.s_b
        )
        residual = corr.weight * (corrected_a - corrected_b)
        jac = corr.weight * (jac_a - jac_b)
        jac_rows.append(jac)
        residual_rows.append(residual)
        meta["corr_slices"][corr.corr_id] = slice(cursor, cursor + 3)
        cursor += 3

    smooth_jac, smooth_res = _smooth_rows(
        layout, params, smooth_weight, gauge_weight
    )
    if smooth_jac.shape[0]:
        jac_rows.append(smooth_jac)
        residual_rows.append(smooth_res)

    if not jac_rows:
        n = layout.n_params
        return np.zeros((0, n)), np.zeros(0), meta
    jacobian = np.vstack(jac_rows)
    residual = np.concatenate(residual_rows)
    return jacobian, residual, meta


def svd_solve(
    jacobian: np.ndarray, residual: np.ndarray, rank_tol: float = RANK_TOL
) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """Solve J dx = -r via SVD. Returns (dx, rank, sv, null_vectors).

    No truncated-rank pseudo-inverse silently projects away data: if the
    rank is deficient the caller receives the null space and the returned
    dx is only the minimum-norm particular solution, clearly labelled.
    """
    if jacobian.shape[0] == 0:
        n = jacobian.shape[1]
        return np.zeros(n), 0, np.zeros(n), np.eye(n)
    u, sv, vt = np.linalg.svd(jacobian, full_matrices=True)
    threshold = rank_tol * float(sv[0]) if sv.size and sv[0] > 0 else rank_tol
    rank = int(np.sum(sv > threshold))
    dx = np.zeros(jacobian.shape[1], dtype=np.float64)
    for i in range(rank):
        dx -= (u[:, i] @ residual) / sv[i] * vt[i]
    null_vectors = vt[rank:].copy() if rank < jacobian.shape[1] else np.zeros((0, jacobian.shape[1]))
    return dx, rank, sv, null_vectors


def _null_labels(vectors: np.ndarray, labels: Sequence[str]) -> List[str]:
    descriptions: List[str] = []
    for vec in vectors:
        order = np.argsort(-np.abs(vec))
        parts = []
        for idx in order[:4]:
            if abs(vec[idx]) < 1e-8:
                break
            parts.append(f"{labels[idx]}({vec[idx]:+.3f})")
        descriptions.append(", ".join(parts) if parts else "(zero vector)")
    return descriptions


def _feasible_candidates(
    layout: ModelLayout,
    params: np.ndarray,
    null_vectors: np.ndarray,
    gcps: Sequence[GcpObs],
    corrs: Sequence[CorrObs],
    smooth_weight: float,
) -> List[dict]:
    """Min-norm solution plus +/- unit steps along each null direction.

    All listed candidates share the same residuals in the row space; they
    are genuinely feasible under the active constraints, which is exactly
    what a deficient-rank problem must expose.
    """
    candidates = [{"name": "minimum-norm particular solution",
                   "params": [float(x) for x in params]}]
    for k, vec in enumerate(null_vectors):
        norm = float(np.linalg.norm(vec))
        if norm < 1e-12:
            continue
        unit = vec / norm
        for sign, tag in ((1.0, "+"), (-1.0, "-")):
            alt = params + sign * CANDIDATE_STEP * unit
            _, residual, _ = _build_system(layout, alt, gcps, corrs, smooth_weight)
            candidates.append({
                "name": f"null dir {k} {tag} {CANDIDATE_STEP:.0f} (param units)",
                "params": [float(x) for x in alt],
                "weighted_residual_norm": float(np.linalg.norm(residual)),
            })
    return candidates


def _apply_update(
    layout: ModelLayout, params: np.ndarray, delta: np.ndarray
) -> np.ndarray:
    new_params = np.asarray(params, dtype=np.float64).copy()
    if layout.model_version == RIGID6:
        current = RigidTransform.from_params(params[:6])
        update = RigidTransform.from_params(delta[:6])
        composed = current.then(update)
        new_params[:6] = composed.to_params()
        return new_params
    if layout.model_version == STRIP_OFFSET3:
        return params + delta
    current = RigidTransform.from_params(params[:6])
    update = RigidTransform.from_params(delta[:6])
    new_params[:6] = current.then(update).to_params()
    new_params[6:] = params[6:] + delta[6:]
    return new_params


def _evaluate(
    layout: ModelLayout,
    params: np.ndarray,
    gcps: Sequence[GcpObs],
    corrs: Sequence[CorrObs],
) -> Tuple[List[dict], List[dict], float]:
    gcp_out: List[dict] = []
    corr_out: List[dict] = []
    squares: List[float] = []
    for gcp in gcps:
        corrected = corrected_point(
            layout, params, gcp.local_xyz, gcp.strip_index, gcp.s
        )
        diff = corrected - gcp.control_xyz
        norm = float(np.linalg.norm(diff))
        gcp_out.append({
            "point_id": gcp.point_id,
            "residual_xyz_m": [float(x) for x in diff],
            "norm_m": norm,
            "weight": gcp.weight,
            "locked": gcp.locked,
        })
        squares.extend([float(x) ** 2 for x in diff])
    for corr in corrs:
        a = corrected_point(layout, params, corr.local_a, corr.strip_a, corr.s_a)
        b = corrected_point(layout, params, corr.local_b, corr.strip_b, corr.s_b)
        diff = a - b
        norm = float(np.linalg.norm(diff))
        corr_out.append({
            "corr_id": corr.corr_id,
            "residual_xyz_m": [float(x) for x in diff],
            "norm_m": norm,
            "weight": corr.weight,
        })
        squares.extend([float(x) ** 2 for x in diff])
    rms = math.sqrt(sum(squares) / len(squares)) if squares else 0.0
    return gcp_out, corr_out, rms


def solve(
    layout: ModelLayout,
    gcps: Sequence[GcpObs],
    corrs: Sequence[CorrObs],
    smooth_weight: float = DEFAULT_SMOOTH_WEIGHT,
    gauge_weight: float = DEFAULT_DRIFT_GAUGE_WEIGHT,
    outlier_sigma: Optional[float] = DEFAULT_OUTLIER_SIGMA,
    reject_outliers: bool = True,
    rank_tol: float = RANK_TOL,
) -> SolveResult:
    """Weighted Gauss-Newton with deterministic correspondence rejection.

    Rejection is deterministic: in each pass the single worst
    over-threshold correspondence is removed and the system is rebuilt from
    scratch.  Rejected correspondences stay in the report with
    ``accepted=false`` as evidence; nothing is deleted.
    """
    active_corrs = list(corrs)
    rejected: List[int] = []
    rank = 0
    sv = np.zeros(layout.n_params)
    null_vectors = np.zeros((0, layout.n_params))
    params = layout.initial_params()
    iterations = 0
    converged = False

    for rejection_pass in range(MAX_REJECT_PASSES + 1):
        params = layout.initial_params()
        iterations = 0
        converged = False
        deficient = False

        for iteration in range(1, MAX_ITERATIONS + 1):
            iterations = iteration
            jacobian, residual, _ = _build_system(
                layout, params, gcps, active_corrs, smooth_weight,
                gauge_weight,
            )
            delta, rank, sv, null_vectors = svd_solve(jacobian, residual, rank_tol)
            if rank < layout.n_params:
                # No unique solution: apply the min-norm particular step once
                # and stop; never reject data to "fix" a geometric deficiency.
                params = _apply_update(layout, params, delta)
                deficient = True
                break
            params = _apply_update(layout, params, delta)
            if float(np.linalg.norm(delta)) < STEP_TOL:
                converged = True
                break

        if deficient:
            break

        _, corr_res, _ = _evaluate(layout, params, gcps, active_corrs)
        if not reject_outliers or outlier_sigma is None:
            break
        over = [c for c in corr_res if c["norm_m"] > outlier_sigma]
        if not over:
            break
        if rejection_pass >= MAX_REJECT_PASSES:
            break
        worst = max(over, key=lambda c: c["norm_m"])
        active_corrs = [c for c in active_corrs if c.corr_id != worst["corr_id"]]
        rejected.append(worst["corr_id"])

    labels = parameter_labels(layout)
    status = "ok"
    message = ""
    candidates: List[dict] = []
    if rank < layout.n_params:
        status = "rank_deficient"
        message = (
            f"rank {rank} / {layout.n_params}; the active constraints leave "
            f"{layout.n_params - rank} unconstrained direction(s); no unique "
            "solution exists, so no regularized solution was chosen."
        )
        candidates = _feasible_candidates(
            layout, params, null_vectors, gcps, active_corrs, smooth_weight
        )
    elif not converged:
        status = "did_not_converge"
        message = f"Gauss-Newton did not converge within {MAX_ITERATIONS} iterations"

    # Re-evaluate including rejected correspondences for the full evidence log.
    all_gcp, all_corr, rms = _evaluate(layout, params, gcps, corrs)
    rejected_set = set(rejected)
    for item in all_corr:
        item["accepted"] = item["corr_id"] not in rejected_set
        if item["corr_id"] in rejected_set:
            item["rejection_reason"] = (
                f"residual {item['norm_m']:.4f} m exceeded outlier threshold"
            )

    return SolveResult(
        converged=converged and status == "ok",
        iterations=iterations,
        rank=rank,
        n_params=layout.n_params,
        singular_values=[float(x) for x in sv],
        params=[float(x) for x in params],
        null_directions=[[float(x) for x in v] for v in null_vectors],
        null_labels=_null_labels(null_vectors, labels),
        candidates=candidates,
        rms_m=rms,
        gcp_residuals=all_gcp,
        corr_residuals=all_corr,
        rejected_corr_ids=rejected,
        status=status,
        message=message,
    )

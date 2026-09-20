"""Rank deficiency, unconstrained directions and rejected-evidence tests."""
from __future__ import annotations

import pytest

import numpy as np

from app.geo.models import build_layout, RIGID6, STRIP_OFFSET3, STRIP_DRIFT
from app.geo.solver import (
    CorrObs,
    GcpObs,
    solve,
)


def test_collinear_gcps_report_rank_and_unconstrained_direction():
    # Three GCPs on the x-axis cannot constrain rotation about that axis.
    line = np.array([[0.0, 0, 0], [10.0, 0, 0], [20.0, 0, 0]])
    shift = np.array([0.1, 0.2, 0.3])
    layout = build_layout(RIGID6, ["S0"])
    gcps = [GcpObs(point_id=i, strip_index=0, s=0.5 * i, local_xyz=line[i], control_xyz=line[i] + shift, weight=10.0) for i in range(3)]
    result = solve(layout, gcps, [], smooth_weight=0.0, reject_outliers=False)
    assert result.status == "rank_deficient"
    assert result.rank == 5
    assert result.n_params == 6
    assert len(result.null_directions) == 1
    vec = np.array(result.null_directions[0])
    # The unconstrained direction must be rotation about x.
    labels = result.null_labels[0]
    assert "global.rx" in labels
    assert abs(vec[3]) == pytest.approx(1.0, abs=1e-6)
    assert np.argmax(np.abs(vec)) == 3
    # Multiple feasible candidates are returned (no silent regularization).
    names = [c["name"] for c in result.candidates]
    assert any("minimum-norm" in n for n in names)
    assert any("null dir 0 +" in n for n in names)
    assert any("null dir 0 -" in n for n in names)


def test_no_constraints_is_fully_degenerate():
    layout = build_layout(STRIP_OFFSET3, ["S0", "S1"])
    result = solve(layout, [], [], smooth_weight=0.0, reject_outliers=False)
    assert result.status == "rank_deficient"
    assert result.rank == 0
    assert len(result.null_directions) == 6
    assert len(result.candidates) >= 3


def test_rejected_correspondence_is_kept_as_evidence():
    layout = build_layout(STRIP_OFFSET3, ["S0", "S1"])
    gcp_local = np.array([[0, 0, 0], [10, 0, 1], [0, 10, 2], [10, 10, 0]], float)
    gcps = [GcpObs(point_id=i, strip_index=0, s=0.3 * i, local_xyz=gcp_local[i], control_xyz=gcp_local[i], weight=10.0) for i in range(4)]
    base = np.array([[5.0, 5.0, 0.0], [2.0, 8.0, 0.0], [8.0, 2.0, 0.0]])
    goods = [
        CorrObs(corr_id=cid, strip_a=0, s_a=0.5, local_a=base[i],
                strip_b=1, s_b=0.5, local_b=base[i].copy())
        for i, cid in enumerate((7, 9, 10))
    ]
    bad = CorrObs(corr_id=8, strip_a=0, s_a=0.5, local_a=base[0],
                  strip_b=1, s_b=0.5,
                  local_b=base[0] + np.array([0.0, 0.0, 3.0]))
    result = solve(layout, gcps, goods + [bad], smooth_weight=0.0, outlier_sigma=0.5)
    assert result.rejected_corr_ids == [8]
    by_id = {c["corr_id"]: c for c in result.corr_residuals}
    # Rejected correspondence is retained, flagged, with a reason.
    assert by_id[8]["accepted"] is False
    assert "exceeded outlier threshold" in by_id[8]["rejection_reason"]
    assert by_id[8]["norm_m"] > 2.9
    for cid in (7, 9, 10):
        assert by_id[cid]["accepted"] is True


def test_rejection_is_deterministic_across_orderings():
    layout = build_layout(STRIP_OFFSET3, ["S0", "S1"])
    gcp_local = np.array([[0, 0, 0], [10, 0, 1], [0, 10, 2], [10, 10, 0]], float)
    gcps = [GcpObs(point_id=i, strip_index=0, s=0.3 * i, local_xyz=gcp_local[i], control_xyz=gcp_local[i], weight=10.0) for i in range(4)]
    good_pts = np.array([[5.0, 5.0, 0.0], [2.0, 8.0, 0.0], [8.0, 2.0, 0.0]])
    goods = [
        CorrObs(corr_id=10 + i, strip_a=0, s_a=0.5, local_a=pt,
                strip_b=1, s_b=0.5, local_b=pt.copy())
        for i, pt in enumerate(good_pts)
    ]
    a = good_pts[0]
    bad2m = CorrObs(corr_id=2, strip_a=0, s_a=0.5, local_a=a, strip_b=1, s_b=0.5,
                    local_b=a + np.array([0.0, 0.0, 2.0]))
    bad3m = CorrObs(corr_id=1, strip_a=0, s_a=0.5, local_a=a, strip_b=1, s_b=0.5,
                    local_b=a + np.array([0.0, 0.0, 3.0]))
    r1 = solve(layout, gcps, goods + [bad3m, bad2m], smooth_weight=0.0, outlier_sigma=0.5)
    r2 = solve(layout, gcps, goods + [bad2m, bad3m], smooth_weight=0.0, outlier_sigma=0.5)
    # Worst (3 m) is always rejected first regardless of input order.
    assert r1.rejected_corr_ids == r2.rejected_corr_ids
    assert r1.rejected_corr_ids[0] == 1


def test_gauge_freedom_without_gauge_observation_is_reported():
    # Drift model with smooth rows but zero gauge leaves a global null mode.
    layout = build_layout(STRIP_DRIFT, ["S0"], {"S0": [0.0, 1.0]})
    pts = [np.array([0.0, 0, 0]), np.array([10.0, 0, 1]),
           np.array([0.0, 10, 2]), np.array([10.0, 10, 0])]
    gcps = [GcpObs(point_id=i, strip_index=0, s=float(i >= 2), local_xyz=pts[i], control_xyz=pts[i], weight=10.0) for i in range(4)]
    result = solve(
        layout, gcps, [], smooth_weight=1.0, gauge_weight=0.0,
        reject_outliers=False,
    )
    assert result.status == "rank_deficient"
    # The gauge null vector moves global translation and all knots together.
    assert len(result.null_directions) >= 1


def test_full_drift_model_is_full_rank_with_gauge():
    layout = build_layout(STRIP_DRIFT, ["S0"], {"S0": [0.0, 0.5, 1.0]})
    pts = [(0.0, [0, 0, 0]), (0.0, [10, 0, 1]), (0.5, [0, 10, 2]),
           (0.5, [10, 10, 0]), (1.0, [0, 0, 3]), (1.0, [10, 5, 2])]
    gcps = []
    for i, (s, p) in enumerate(pts):
        p = np.array(p, float)
        gcps.append(GcpObs(point_id=i, strip_index=0, s=s, local_xyz=p, control_xyz=p + np.array([0.1 * s, 0, 0]), weight=10.0))
    result = solve(layout, gcps, [], smooth_weight=1.0, reject_outliers=False)
    assert result.status == "ok"
    assert result.rank == result.n_params

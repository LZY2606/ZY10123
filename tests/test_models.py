"""Segment endpoint ownership and model evaluation."""
from __future__ import annotations

import numpy as np
import pytest

from app.geo.models import build_layout, STRIP_DRIFT, STRIP_OFFSET3, RIGID6, corrected_point
from app.geo.solver import solve, GcpObs, CorrObs


def test_endpoint_belongs_to_exactly_one_earlier_segment():
    layout = build_layout(STRIP_DRIFT, ["A"], {"A": [0.0, 0.5, 1.0]})
    dl = layout.drift["A"]
    # s == 0.5 selects the earlier segment [0, 0.5] with u == 1.
    assert dl.segment_at(0.5) == (0, 0, 1, 1.0)
    # Just above belongs to the later segment.
    seg, left, right, u = dl.segment_at(0.5000001)
    assert (seg, left, right) == (1, 1, 2)
    assert 0.0 < u < 1.0


def test_endpoint_point_loads_only_one_knot():
    layout = build_layout(STRIP_DRIFT, ["A"], {"A": [0.0, 0.5, 1.0]})
    from app.geo.solver import point_jacobian
    params = np.zeros(layout.n_params)
    i_mid = layout.drift_knot_index("A", 1)
    _, jac = point_jacobian(layout, params, np.zeros(3), 0, 0.5)
    # At s = 0.5 exactly, only knot 1 is blended (full weight).
    assert np.allclose(jac[:, i_mid:i_mid + 3], np.eye(3))
    # Knot 0 must receive no load from the endpoint point.
    i_zero = layout.drift_knot_index("A", 0)
    assert np.allclose(jac[:, i_zero:i_zero + 3], 0.0)


def test_endpoint_correction_is_exactly_the_knot_value():
    layout = build_layout(STRIP_DRIFT, ["A"], {"A": [0.0, 0.5, 1.0]})
    params = np.zeros(layout.n_params)
    i_mid = layout.drift_knot_index("A", 1)
    params[i_mid:i_mid + 3] = [0.7, -0.4, 0.2]
    out = corrected_point(layout, params, np.zeros(3), 0, 0.5)
    np.testing.assert_allclose(out, [0.7, -0.4, 0.2], atol=1e-12)


def test_breaks_must_be_valid():
    with pytest.raises(ValueError):
        build_layout(STRIP_DRIFT, ["A"], {"A": [0.1, 1.0]})
    with pytest.raises(ValueError):
        build_layout(STRIP_DRIFT, ["A"], {"A": [0.0, 0.5, 0.5, 1.0]})


def test_offset_model_recovers_known_transforms():
    shift0 = np.array([0.30, -0.15, 0.08])
    shift1 = np.array([-0.10, 0.20, -0.05])
    layout = build_layout(STRIP_OFFSET3, ["S0", "S1"])
    gcp_local = np.array([[0, 0, 0], [10, 0, 1], [0, 10, 2], [10, 10, 0]], float)
    gcps = [GcpObs(point_id=i, strip_index=0, s=0.3 * i,
                   local_xyz=gcp_local[i], control_xyz=gcp_local[i] + shift0,
                   weight=10.0)
            for i in range(4)]
    a = np.array([[2, 2, 0], [8, 2, 0], [2, 8, 0]], float)
    b = a + shift0 - shift1
    corrs = [CorrObs(corr_id=i, strip_a=0, s_a=0.2, local_a=a[i], strip_b=1, s_b=0.2, local_b=b[i]) for i in range(3)]
    result = solve(layout, gcps, corrs, smooth_weight=0.0, reject_outliers=False)
    assert result.status == "ok"
    np.testing.assert_allclose(np.array(result.params[:3]), shift0, atol=1e-8)
    np.testing.assert_allclose(np.array(result.params[3:]), shift1, atol=1e-8)


def test_rigid6_recovers_known_pose():
    truth = np.array([0.25, -0.4, 0.1, 0.03, -0.02, 0.04])
    rng = np.random.default_rng(7)
    from app.geo.transforms import RigidTransform
    local = rng.uniform(-10, 10, size=(8, 3))
    transform_corrected = RigidTransform.from_params(truth).apply(local)
    layout = build_layout(RIGID6, ["S0"])
    gcps = [GcpObs(point_id=i, strip_index=0, s=i / 7, local_xyz=local[i], control_xyz=transform_corrected[i], weight=10.0)
            for i in range(8)]
    result = solve(layout, gcps, [], smooth_weight=0.0, reject_outliers=False)
    assert result.status == "ok"
    assert result.rank == 6
    np.testing.assert_allclose(np.array(result.params), truth, atol=1e-7)


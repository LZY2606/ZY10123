"""Direction-of-composition tests: no left/right-multiplication ambiguity."""
from __future__ import annotations

import numpy as np

from app.geo.transforms import RigidTransform, compose_chain


def test_apply_direction_is_source_to_target():
    # Pure translation: target = source + t (documented convention).
    t = RigidTransform.from_params([1.0, 2.0, 3.0, 0, 0, 0])
    p = np.array([[0.0, 0.0, 0.0]])
    out = t.apply(p)
    np.testing.assert_allclose(out, [[1.0, 2.0, 3.0]], atol=1e-12)


def test_then_composes_in_execution_order():
    # T_ab: a -> b translates +x by 1; T_bc: b -> c translates +y by 2.
    # Include rotations so that execution order genuinely matters.
    tab = RigidTransform.from_params([1, 0, 0, 0.0, 0.0, 0.2])
    tbc = RigidTransform.from_params([0, 2, 0, 0.0, 0.3, 0.0])
    p_a = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]])
    expected_a = (RigidTransform.from_params([0, 0, 0, 0, 0, 0.2])
                  .then(RigidTransform.from_params([1, 0, 0, 0, 0, 0]))
                  .then(RigidTransform.from_params([0, 0, 0, 0, 0.3, 0]))
                  .then(RigidTransform.from_params([0, 2, 0, 0, 0, 0])))
    p_c = tab.then(tbc).apply(p_a)
    np.testing.assert_allclose(
        p_c, expected_a.apply(p_a), atol=1e-12
    )
    # The reverse execution order gives a different point; assert we did
    # not silently swap operand order.
    p_other = tbc.then(tab).apply(p_a)
    assert not np.allclose(p_c, p_other)


def test_inverse_roundtrip_with_large_rotation():
    rng = np.random.default_rng(42)
    params = np.array([0.4, -0.7, 0.2, 0.6, -0.5, 0.9])
    t = RigidTransform.from_params(params)
    points = rng.normal(size=(20, 3))
    back = t.inverse().apply(t.apply(points))
    np.testing.assert_allclose(back, points, atol=1e-10)
    # Inverse rotation is orthonormal, not a least-squares approximation.
    np.testing.assert_allclose(t.inverse().rotation @ t.rotation, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(t.rotation), 1.0, atol=1e-12)


def test_chain_then_inverse_is_identity():
    t1 = RigidTransform.from_params([0.1, 0.2, -0.3, 0.2, 0.1, -0.4])
    t2 = RigidTransform.from_params([-0.5, 0.0, 0.2, -0.3, 0.4, 0.1])
    chain = compose_chain([t1, t2])
    back = compose_chain([chain, chain.inverse()])
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(10, 3))
    np.testing.assert_allclose(back.apply(pts), pts, atol=1e-10)


def test_local_twist_stacking_stays_on_manifold():
    # Repeated solver-style composition must remain an exact rigid motion.
    cur = RigidTransform.identity()
    step = RigidTransform.from_params([0.01, -0.01, 0.005, 0.02, 0.01, -0.02])
    for _ in range(50):
        cur = cur.then(step)
    np.testing.assert_allclose(cur.rotation.T @ cur.rotation, np.eye(3), atol=1e-12)

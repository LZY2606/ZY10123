"""Rigid (SE(3)) transforms with an explicit direction convention.

Convention used everywhere in this project
------------------------------------------
A transform ``T`` maps coordinates expressed in the *source* frame to
coordinates expressed in the *target* frame::

    p_target = T.apply(p_source)

* Composition ``T_ab.then(T_bc)`` means: first ``T_ab`` (source a -> b),
  then ``T_bc`` (b -> c); the result maps a -> c.  Transform application
  and composition are written in execution order, never in matrix
  left/right multiplication order, so the chain cannot silently flip.
* ``T.inverse()`` performs the exact inverse (orthonormal rotation), not a
  least-squares "approximately inverse" matrix.

Parameters (used by the solver) are ``[tx, ty, tz, rx, ry, rz]`` where the
rotation components are Rodrigues parameters (axis * angle).  The Lie-group
composition is used when stacking solver updates, so repeated updates remain
exactly on SE(3).

Angles are stored/serialized in **radians**, lengths in **metres**.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

DIM = 3
RODRIGUES_EPS = 1e-12


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = (float(v[0]), float(v[1]), float(v[2]))
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64
    )


def rodrigues_rotation(rvec: Sequence[float]) -> np.ndarray:
    """Rotation matrix for Rodrigues parameters r = axis * angle (rad)."""
    r = np.asarray(rvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(r))
    if theta < RODRIGUES_EPS:
        # First-order expansion, deterministic at zero.
        return np.eye(3, dtype=np.float64) + skew(r)
    axis = r / theta
    k = skew(axis)
    c = math.cos(theta)
    s = math.sin(theta)
    rot = (
        np.eye(3, dtype=np.float64) * c
        + (1.0 - c) * np.outer(axis, axis)
        + s * k
    )
    return rot


def rotation_log(rot: np.ndarray) -> np.ndarray:
    """Inverse of :func:`rodrigues_rotation`: log of a rotation matrix."""
    r = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    cos_angle = (np.trace(r) - 1.0) / 2.0
    cos_angle = min(1.0, max(-1.0, cos_angle))
    theta = math.acos(cos_angle)
    if theta < 1e-10:
        return 0.5 * np.array(
            [r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]],
            dtype=np.float64,
        )
    if abs(theta - math.pi) < 1e-9:
        # 180 degree branch: axis from the diagonal.
        diag = np.clip((np.diag(r) + 1.0) / 2.0, 0.0, 1.0)
        axis = np.sqrt(diag)
        # Resolve signs from off-diagonal products.
        if axis[0] >= axis[1] and axis[0] >= axis[2]:
            axis = np.array(
                [axis[0], np.sign(r[0, 1] or 1.0) * axis[1],
                 np.sign(r[0, 2] or 1.0) * axis[2]]
            )
        elif axis[1] >= axis[2]:
            axis = np.array(
                [np.sign(r[1, 0] or 1.0) * axis[0], axis[1],
                 np.sign(r[1, 2] or 1.0) * axis[2]]
            )
        else:
            axis = np.array(
                [np.sign(r[2, 0] or 1.0) * axis[0],
                 np.sign(r[2, 1] or 1.0) * axis[1], axis[2]]
            )
        return axis * theta
    factor = theta / (2.0 * math.sin(theta))
    return factor * np.array(
        [r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class RigidTransform:
    """SE(3) transform.  ``apply(p) = R @ p + t``; maps source -> target."""

    translation: np.ndarray
    rotation: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "translation", np.asarray(self.translation, dtype=np.float64).reshape(3)
        )
        object.__setattr__(
            self, "rotation", np.asarray(self.rotation, dtype=np.float64).reshape(3, 3)
        )

    @staticmethod
    def identity() -> "RigidTransform":
        return RigidTransform(np.zeros(3), np.eye(3))

    @staticmethod
    def from_params(params: Sequence[float]) -> "RigidTransform":
        p = np.asarray(params, dtype=np.float64).reshape(6)
        return RigidTransform(p[:3].copy(), rodrigues_rotation(p[3:6]))

    def to_params(self) -> np.ndarray:
        return np.concatenate([self.translation, rotation_log(self.rotation)])

    def apply(self, points: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        return pts @ self.rotation.T + self.translation

    def then(self, other: "RigidTransform") -> "RigidTransform":
        """Return ``other o self``: execute self first, then other."""
        return RigidTransform(
            other.rotation @ self.translation + other.translation,
            other.rotation @ self.rotation,
        )

    def inverse(self) -> "RigidTransform":
        rt = self.rotation.T
        return RigidTransform(-(rt @ self.translation), rt)

    def to_matrix(self) -> np.ndarray:
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = self.rotation
        mat[:3, 3] = self.translation
        return mat

    def to_json(self) -> dict:
        return {
            "translation": [float(x) for x in self.translation],
            "rotation_rvec_rad": [float(x) for x in rotation_log(self.rotation)],
            "direction": "source -> target (p_target = R p_source + t)",
        }


def compose_chain(transforms: Iterable[RigidTransform]) -> RigidTransform:
    """Compose transforms in execution order: T0 then T1 then ..."""
    result = RigidTransform.identity()
    for transform in transforms:
        result = result.then(transform)
    return result


def params_to_transform_delta(params: Sequence[float]) -> RigidTransform:
    """Interpret a 6-vector as a small SE(3) update (Rodrigues params)."""
    return RigidTransform.from_params(params)

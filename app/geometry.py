"""Rigid transformations with one documented composition direction.

All storage and API residuals use this direction::

    common_point = R @ local_point + segment_translation

A common-to-local transform therefore uses the inverse rotation and inverse
translation.  Callers never pass a matrix whose intended side is ambiguous.
"""
from __future__ import annotations

import math

Vector3 = tuple[float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]

IDENTITY: Matrix3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
ZERO: Vector3 = (0.0, 0.0, 0.0)


def add(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def matvec(a: Matrix3, v: Vector3) -> Vector3:
    return (
        a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
        a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
        a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
    )


def matmul(a: Matrix3, b: Matrix3) -> Matrix3:
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )  # type: ignore[return-value]


def transpose(a: Matrix3) -> Matrix3:
    return tuple(tuple(row) for row in zip(*a))  # type: ignore[return-value]


def skew(v: Vector3) -> Matrix3:
    return (
        (0.0, -v[2], v[1]),
        (v[2], 0.0, -v[0]),
        (-v[1], v[0], 0.0),
    )


def rotation_xyz(angles: Vector3) -> Matrix3:
    """Rotation for intrinsic-looking Rz @ Ry @ Rx parameter angles."""
    cx, sx = math.cos(angles[0]), math.sin(angles[0])
    cy, sy = math.cos(angles[1]), math.sin(angles[1])
    cz, sz = math.cos(angles[2]), math.sin(angles[2])
    rx = ((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx))
    ry = ((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy))
    rz = ((cz, -sz, 0.0), (sz, cz, 0.0), (0.0, 0.0, 1.0))
    return matmul(matmul(rz, ry), rx)


def exp_so3(omega: Vector3) -> Matrix3:
    angle = math.sqrt(sum(value * value for value in omega))
    if angle < 1e-14:
        axis_skew = skew(omega)
        return tuple(
            tuple(IDENTITY[i][j] + axis_skew[i][j] for j in range(3))
            for i in range(3)
        )  # type: ignore[return-value]
    axis = (omega[0] / angle, omega[1] / angle, omega[2] / angle)
    k = skew(axis)
    k2 = matmul(k, k)
    sine = math.sin(angle)
    cosine = math.cos(angle)
    return tuple(
        tuple(
            IDENTITY[i][j] + sine * k[i][j] + (1.0 - cosine) * k2[i][j]
            for j in range(3)
        )
        for i in range(3)
    )  # type: ignore[return-value]


def inverse_transform(rotation: Matrix3, translation: Vector3) -> tuple[Matrix3, Vector3]:
    """Inverse of common = R local + t, applied in the opposite direction."""
    inv_r = transpose(rotation)
    return inv_r, tuple(-value for value in matvec(inv_r, translation))  # type: ignore[return-value]


def apply_local_to_common(
    point: Vector3,
    rotation: Matrix3,
    translation: Vector3,
    drift: Vector3 = ZERO,
) -> Vector3:
    return add(matvec(rotation, add(point, drift)), translation)


def apply_common_to_local(
    point: Vector3,
    rotation: Matrix3,
    translation: Vector3,
    drift: Vector3 = ZERO,
) -> Vector3:
    inv_r, _ = inverse_transform(rotation, translation)
    return sub(matvec(inv_r, sub(point, translation)), drift)


def segment_index(time_value: float, boundaries: list[float]) -> int:
    """Return the left-closed/right-open segment index.

    A time exactly on a boundary belongs to the following segment.  The final
    segment closes at the strip's recorded end time.
    """
    ordered = sorted(boundaries)
    for index, boundary in enumerate(ordered):
        if time_value < boundary:
            return index
    return len(ordered)

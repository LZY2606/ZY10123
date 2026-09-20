"""Deterministic rank-revealing linear algebra used by the adjustment core.

The implementation deliberately has no randomised solver path.  Free variables
are reported instead of being silently stabilised with a regulariser.
"""
from __future__ import annotations

import math
from typing import Sequence

Matrix = list[list[float]]
Vector = list[float]


def matvec(a: Matrix, x: Vector) -> Vector:
    return [sum(row[j] * x[j] for j in range(len(x))) for row in a]


def matmul(a: Matrix, b: Matrix) -> Matrix:
    cols = len(b[0])
    inner = len(b)
    return [
        [sum(a[i][k] * b[k][j] for k in range(inner)) for j in range(cols)]
        for i in range(len(a))
    ]


def transpose(a: Matrix) -> Matrix:
    return [list(row) for row in zip(*a)]


def norm(v: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in v))


def normalize(v: Vector) -> Vector:
    length = norm(v)
    if length <= 0.0:
        raise ValueError("cannot normalize a zero vector")
    return [value / length for value in v]


def rank_revealing_least_squares(
    design: Matrix,
    response: Vector,
    tolerance: float = 1e-10,
    cols: int | None = None,
) -> dict:
    """Solve Ax = b by Householder QR with column pivoting.

    The returned particular solution sets unresolved free parameters to zero.
    That choice is labelled and accompanied by the null-space basis; callers
    must not present it as a unique physical solution.
    """
    if not design:
        width = 0 if cols is None else cols
        return {
            "status": "ok",
            "rank": 0,
            "degrees_of_freedom": width,
            "observations": 0,
            "solution": [0.0] * width,
            "null_basis": [[1.0 if j == i else 0.0 for j in range(width)] for i in range(width)],
            "residual": response,
            "residual_norm": norm(response),
        }

    rows = len(design)
    if cols is None:
        cols = len(design[0])
    elif cols != len(design[0]):
        raise ValueError("explicit column count must match the design matrix")
    qr = [row[:] for row in design]
    permutation = list(range(cols))
    rhs = response[:]
    householder: list[tuple[int, Vector, float]] = []

    for k in range(min(rows, cols)):
        squared = [0.0] * cols
        for j in range(k, cols):
            squared[j] = sum(qr[i][j] * qr[i][j] for i in range(k, rows))
        pivot = max(range(k, cols), key=lambda j: squared[j])
        if squared[pivot] <= tolerance * tolerance:
            break
        if pivot != k:
            for row in qr:
                row[k], row[pivot] = row[pivot], row[k]
            permutation[k], permutation[pivot] = permutation[pivot], permutation[k]

        column = [qr[i][k] for i in range(k, rows)]
        alpha = norm(column)
        if column[0] >= 0.0:
            alpha = -alpha
        unit = column[:]
        unit[0] -= alpha
        unit = normalize(unit)
        scale = 2.0
        householder.append((k, unit, scale))

        for j in range(k, cols):
            dot = sum(unit[r] * qr[k + r][j] for r in range(len(unit)))
            factor = scale * dot
            for r in range(len(unit)):
                qr[k + r][j] -= factor * unit[r]
        dot = sum(unit[r] * rhs[k + r] for r in range(len(unit)))
        factor = scale * dot
        for r in range(len(unit)):
            rhs[k + r] -= factor * unit[r]

    diag = [abs(qr[k][k]) if k < min(rows, cols) else 0.0 for k in range(cols)]
    scale = max(diag + [1.0])
    rank = sum(1 for value in diag if value > tolerance * scale)

    ordered = [0.0] * cols
    for i in range(rank - 1, -1, -1):
        value = rhs[i]
        for j in range(i + 1, rank):
            value -= qr[i][j] * ordered[j]
        if abs(qr[i][i]) <= tolerance * scale:
            raise RuntimeError("rank-revealing QR encountered a singular pivot")
        ordered[i] = value / qr[i][i]

    null_basis = []
    for free in range(rank, cols):
        vector = [0.0] * cols
        vector[free] = 1.0
        for i in range(rank - 1, -1, -1):
            value = -sum(qr[i][j] * vector[j] for j in range(i + 1, cols))
            vector[i] = value / qr[i][i]
        null_basis.append(vector)

    solution = [0.0] * cols
    for position, parameter in enumerate(permutation):
        solution[parameter] = ordered[position]
    basis = []
    for vector in null_basis:
        restored = [0.0] * cols
        for position, parameter in enumerate(permutation):
            restored[parameter] = vector[position]
        basis.append(normalize(restored))

    residual = [
        response[row] - sum(design[row][j] * solution[j] for j in range(cols))
        for row in range(rows)
    ]
    return {
        "status": "ok",
        "rank": rank,
        "degrees_of_freedom": cols,
        "observations": rows,
        "solution": solution,
        "null_basis": basis,
        "residual": residual,
        "residual_norm": norm(residual),
    }


def constrained_least_squares(
    design: Matrix,
    response: Vector,
    equality: Matrix | None,
    equality_rhs: Vector | None,
    tolerance: float = 1e-10,
) -> dict:
    """Weighted linear least squares with exact linear equality constraints."""
    cols = len(design[0]) if design else (len(equality[0]) if equality else 0)
    if not equality:
        return rank_revealing_least_squares(design, response, tolerance, cols)

    m = len(equality)
    c = [row[:] for row in equality]
    d = equality_rhs[:]
    pivot_rows: list[int] = []
    pivot_cols: list[int] = []
    free_cols = list(range(cols))

    for _ in range(min(m, cols)):
        candidate = max(
            (
                abs(c[r][col])
                for r in range(m)
                if r not in pivot_rows
                for col in free_cols
            ),
            default=0.0,
        )
        if candidate <= tolerance:
            break
        row = next(
            r
            for r in range(m)
            if r not in pivot_rows
            for col in free_cols
            if abs(c[r][col]) > tolerance
        )
        col = next(col for col in free_cols if abs(c[row][col]) > tolerance)
        pivot_rows.append(row)
        pivot_cols.append(col)
        free_cols.remove(col)
        pivot_value = c[row][col]
        for j in range(cols):
            c[row][j] /= pivot_value
        d[row] /= pivot_value
        for r in range(m):
            if r == row:
                continue
            factor = c[r][col]
            if abs(factor) <= tolerance:
                continue
            for j in range(cols):
                c[r][j] -= factor * c[row][j]
            d[r] -= factor * d[row]

    for r in range(m):
        if r in pivot_rows:
            continue
        if abs(d[r]) > 1e-8 or any(abs(value) > tolerance for value in c[r]):
            return {
                "status": "inconsistent_constraints",
                "rank": len(pivot_cols),
                "degrees_of_freedom": cols,
                "null_basis": [],
                "solution": [],
                "constraint_residual": d,
            }

    particular = [0.0] * cols
    for row, col in zip(pivot_rows, pivot_cols):
        particular[col] = d[row]

    basis = []
    for free in free_cols:
        vector = [0.0] * cols
        vector[free] = 1.0
        for row, col in zip(pivot_rows, pivot_cols):
            vector[col] = -sum(c[row][j] * vector[j] for j in range(cols) if j != col)
        basis.append(vector)

    if basis:
        z = transpose(basis)
        reduced_a = matmul(design, z) if design else []
        reduced_b = [
            response[i] - sum(design[i][j] * particular[j] for j in range(cols))
            for i in range(len(design))
        ] if design else []
        result = rank_revealing_least_squares(reduced_a, reduced_b, tolerance)
        if result["status"] == "inconsistent_constraints":
            return result
        free_solution = result["solution"]
        solution = [
            particular[j] + sum(basis[k][j] * free_solution[k] for k in range(len(basis)))
            for j in range(cols)
        ]
        mapped_null = [
            [sum(basis[k][j] * value[k] for k in range(len(basis))) for j in range(cols)]
            for value in result["null_basis"]
        ]
        rank = len(pivot_cols) + result["rank"]
    else:
        solution = particular
        mapped_null = []
        rank = cols

    residual = [
        response[i] - sum(design[i][j] * solution[j] for j in range(cols))
        for i in range(len(design))
    ] if design else []
    constraint_residual = [
        equality_rhs[r] - sum(equality[r][j] * solution[j] for j in range(cols))
        for r in range(m)
    ]
    return {
        "status": "ok",
        "rank": rank,
        "degrees_of_freedom": cols,
        "observations": len(design),
        "solution": solution,
        "null_basis": mapped_null,
        "residual": residual,
        "residual_norm": norm(residual),
        "constraint_residual": constraint_residual,
        "equality_rank": len(pivot_cols),
    }

"""Alignment models for LiDAR strips.

Model versions (frozen once published; changing a layout bumps the version):

* ``rigid6-v1``  : one global 6-DoF transform for every strip.
* ``strip_offset3-v1`` : one 3-component translation (tx, ty, tz) per strip.
* ``strip_drift-v1`` : per strip, piecewise-linear translation drift along
  the acquisition path, plus one global small rotation (rx, ry, rz) shared
  by all strips.  With knots ``b_0=0 < b_1 < ... b_{n-1}=1`` the correction
  at normalized along-track position ``s`` is

      corr(s) = (1 - u) * d_left + u * d_right,   u = (s-b_k)/(b_{k+1}-b_k)

  where ``[b_k, b_{k+1})`` is the containing segment.

Endpoint ownership rule (single-segment membership)
----------------------------------------------------
For an interior knot ``b_k``, a point with ``s == b_k`` belongs **only to
the left/earlier segment** and receives exactly ``d_k``::

    j = max(bisect_right(breaks, s) - 1, 0), clipped to [0, n_segments-1]

so ``s = b_k`` selects segment ``k-1`` with ``u = 1``; it is never blended
twice and never also belongs to segment ``k``.

Direction convention (identical for every model): local strip frame
(source) -> common control frame (target), i.e. ``p_control = model(p)``.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .transforms import RigidTransform

RIGID6 = "rigid6-v1"
STRIP_OFFSET3 = "strip_offset3-v1"
STRIP_DRIFT = "strip_drift-v1"
MODEL_VERSIONS = (RIGID6, STRIP_OFFSET3, STRIP_DRIFT)


@dataclass(frozen=True)
class DriftLayout:
    strip_id: str
    breaks: Tuple[float, ...]

    @property
    def n_knots(self) -> int:
        return len(self.breaks)

    @property
    def n_segments(self) -> int:
        return len(self.breaks) - 1

    def segment_at(self, s: float) -> Tuple[int, int, int, float]:
        """Return (segment_index, left_knot, right_knot, blend u).

        Exact interior-break points belong to the left segment (u == 1).
        Values outside [0, 1] clamp to the first/last segment.
        """
        breaks = list(self.breaks)
        if s <= 0.0:
            return 0, 0, 1, 0.0
        if s >= 1.0:
            j = self.n_segments - 1
            return j, j, j + 1, 1.0
        # bisect_left: s exactly at interior knot b_k selects segment k-1
        # (the earlier segment) with u == 1, so the point loads only knot k.
        j = min(max(bisect.bisect_left(breaks, s) - 1, 0), self.n_segments - 1)
        left, right = j, j + 1
        span = breaks[right] - breaks[left]
        u = 0.0 if span <= 0.0 else (s - breaks[left]) / span
        return j, left, right, float(np.clip(u, 0.0, 1.0))


def normalize_breaks(breaks: Sequence[float]) -> Tuple[float, ...]:
    arr = [float(x) for x in breaks]
    if len(arr) < 2:
        raise ValueError("drift model needs at least two knots [0.0, 1.0]")
    if arr[0] != 0.0 or arr[-1] != 1.0:
        raise ValueError("drift breaks must start at 0.0 and end at 1.0")
    for a, b in zip(arr, arr[1:]):
        if b <= a:
            raise ValueError("drift breaks must be strictly increasing")
    return tuple(arr)


@dataclass
class ModelLayout:
    """Parameter-block layout for one alignment configuration."""

    model_version: str
    strip_ids: Tuple[str, ...]
    drift: Dict[str, DriftLayout]

    @property
    def strips(self) -> Tuple[str, ...]:
        return self.strip_ids

    @property
    def n_params(self) -> int:
        if self.model_version == RIGID6:
            return 6
        if self.model_version == STRIP_OFFSET3:
            return 3 * len(self.strip_ids)
        count = 6
        for sid in self.strip_ids:
            count += 3 * self.drift[sid].n_knots
        return count

    def strip_offset_index(self, strip_id: str) -> int:
        return 3 * self.strip_ids.index(strip_id)

    def drift_knot_index(self, strip_id: str, knot: int) -> int:
        base = 6
        order = self.strip_ids.index(strip_id)
        for sid in self.strip_ids[:order]:
            base += 3 * self.drift[sid].n_knots
        return base + 3 * knot

    def initial_params(self) -> np.ndarray:
        return np.zeros(self.n_params, dtype=np.float64)


def build_layout(
    model_version: str,
    strip_ids: Sequence[str],
    drift_breaks: Optional[Dict[str, Sequence[float]]] = None,
) -> ModelLayout:
    if model_version not in MODEL_VERSIONS:
        raise ValueError(f"unknown model version: {model_version}")
    ids = tuple(strip_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate strip ids")
    layouts: Dict[str, DriftLayout] = {}
    if model_version == STRIP_DRIFT:
        for sid in ids:
            raw = (drift_breaks or {}).get(sid, [0.0, 1.0])
            layouts[sid] = DriftLayout(sid, normalize_breaks(raw))
    return ModelLayout(model_version, ids, layouts)


def corrected_point(
    layout: ModelLayout,
    params: np.ndarray,
    point: np.ndarray,
    strip_index: int,
    s: float,
) -> np.ndarray:
    p = np.asarray(params, dtype=np.float64)
    point = np.asarray(point, dtype=np.float64)
    if layout.model_version == RIGID6:
        return RigidTransform.from_params(p[:6]).apply(point.reshape(1, 3))[0]
    if layout.model_version == STRIP_OFFSET3:
        j = 3 * strip_index
        return point + p[j:j + 3]
    sid = layout.strip_ids[strip_index]
    _, left, right, u = layout.drift[sid].segment_at(float(s))
    transform = RigidTransform.from_params(p[:6])
    il = layout.drift_knot_index(sid, left)
    ir = layout.drift_knot_index(sid, right)
    drift = (1.0 - u) * p[il:il + 3] + u * p[ir:ir + 3]
    return transform.apply(point.reshape(1, 3))[0] + drift


def evaluate_points(
    layout: ModelLayout,
    params: np.ndarray,
    points: np.ndarray,
    strip_indices: np.ndarray,
    along_s: np.ndarray,
) -> np.ndarray:
    """Apply the model: local strip frame (source) -> control frame."""
    pts = np.asarray(points, dtype=np.float64)
    indices = np.asarray(strip_indices)
    s_values = np.asarray(along_s, dtype=np.float64)
    out = np.empty_like(pts)
    for row in range(pts.shape[0]):
        out[row] = corrected_point(layout, params, pts[row], int(indices[row]), float(s_values[row]))
    return out

"""Deterministic synthetic demo dataset.

Geometry (all local coordinates stored verbatim in metres)
----------------------------------------------------------
* Common control frame.  Strip S1 local frame equals the control frame plus
  a known misregistration ``shift1``; strip S2 local frame plus ``shift2``.
* Four non-collinear GCPs lie on S1; their published control coordinates
  are their true control-frame positions.
* Overlap correspondences join points that coincide in the control frame
  (so after recovering the shifts their residuals are ~0).
* One planted gross outlier correspondence (3 m in z).

``unit`` / ``unit_scale_to_m`` = ``m`` / 1.0; a second demo variant is not
needed because the import API accepts ``cm``/``ft`` units for real uploads.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

SHIFT1 = (0.30, -0.15, 0.08)
SHIFT2 = (-0.10, 0.20, -0.05)


def demo_payload() -> Dict[str, Any]:
    # True control-frame positions for shared features.
    shared = [
        (20.0, 0.0, 0.30),
        (40.0, 2.0, 0.45),
        (60.0, -1.0, 0.30),
        (80.0, 1.0, 0.10),
    ]

    s1_points: List[Dict[str, Any]] = []
    s2_points: List[Dict[str, Any]] = []
    # S1 covers control y 0..100; S2 covers control y 0..100 as well, with a
    # slight local x bias. Local = control - strip_shift.
    for k in range(11):
        cy = k * 10.0
        cx = 0.0 if k % 2 == 0 else 3.0
        cz = 0.4 * math.sin(k / 2.0)
        s1_points.append({
            "id": 100 + k,
            "x": round(cx - SHIFT1[0], 6),
            "y": round(cy - SHIFT1[1], 6),
            "z": round(cz - SHIFT1[2], 6),
            "timestamp_s": float(k),
        })
        s2_points.append({
            "id": 200 + k,
            # Same control-frame footprint as S1 (overlapping strips); only
            # the local-frame misregistration SHIFT2 differs.
            "x": round(cx - SHIFT2[0], 6),
            "y": round(cy - SHIFT2[1], 6),
            "z": round(cz - SHIFT2[2], 6),
            "timestamp_s": float(k),
        })

    # GCPs on S1 at control rows k=0,2,7,10 (non-collinear in x/y and spanning z).
    gcp_rows = [(0, "G1"), (2, "G2"), (7, "G3"), (10, "G4")]
    gcps: List[Dict[str, Any]] = []
    for k, code in gcp_rows:
        cy = k * 10.0
        cx = 0.0 if k % 2 == 0 else 3.0
        cz = 0.4 * math.sin(k / 2.0)
        gcps.append({
            "point_id": 100 + k, "code": code,
            "control_x": round(cx, 6), "control_y": round(cy, 6),
            "control_z": round(cz, 6),
            "control_unit": "m", "control_crs": "demo-control-local-m",
        })

    # Overlap correspondences: shared control features.
    # S1 point index k has control y = 10k; shared[i] has control y at rows
    # (2,4,6,8) in both strips; local = control - shift.
    correspondences: List[Dict[str, Any]] = []
    for i, (cy, _dx, _cz) in enumerate(shared):
        k = int(round(cy / 10.0))
        correspondences.append({"point_a_id": 100 + k, "point_b_id": 200 + k})
    # Planted outlier: S1 k=0 vs S2 k=9 (gross mismatch).
    correspondences.append({"point_a_id": 100, "point_b_id": 209})

    return {
        "name": "demo two-strip overlap",
        "unit": "m",
        "crs": "demo-local-m",
        "source_description": (
            "Synthetic deterministic demo. Stored coordinates are local "
            "metres verbatim; control frame is demo-control-local-m. "
            "Normalization applies the unit scale only (m -> 1.0)."
        ),
        "strips": [
            {"strip_id": "S1", "points": s1_points},
            {"strip_id": "S2", "points": s2_points},
        ],
        "gcps": gcps,
        "correspondences": correspondences,
    }

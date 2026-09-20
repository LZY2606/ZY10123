from __future__ import annotations

from .database import many
from .services import import_dataset


def demo_request() -> dict:
    return {
        "job_id": "job-import-demo-v2",
        "name": "Two-strip LiDAR control demo",
        "source_name": "synthetic_feature_extract_v1",
        "common_crs": "local engineering CRS / common demo datum",
        "common_length_unit": "m",
        "metadata": {"point_fidelity": "coordinates imported verbatim; no LAS/LAZ processing"},
        "strips": [
            {"strip_id": "S1", "label": "Strip one", "time_start": 0.0, "time_end": 10.0,
             "time_unit": "s", "time_epoch": "demo_clock", "local_crs": "S1 local engineering CRS",
             "local_length_unit": "m"},
            {"strip_id": "S2", "label": "Strip two", "time_start": 0.0, "time_end": 10.0,
             "time_unit": "s", "time_epoch": "demo_clock", "local_crs": "S2 local engineering CRS",
             "local_length_unit": "m"},
        ],
        "points": [
            {"point_id": "p1", "strip_id": "S1", "time_value": 0.0, "x": 0.0, "y": 0.0, "z": 0.0},
            {"point_id": "p2", "strip_id": "S1", "time_value": 5.0, "x": 5.0, "y": 0.0, "z": 1.0},
            {"point_id": "p3", "strip_id": "S1", "time_value": 10.0, "x": 10.0, "y": 0.0, "z": 2.0},
            {"point_id": "p4", "strip_id": "S1", "time_value": 5.0, "x": 5.0, "y": 4.0, "z": 1.0},
            {"point_id": "p5", "strip_id": "S1", "time_value": 5.0, "x": 0.0, "y": 4.0, "z": 2.0},
            {"point_id": "q1", "strip_id": "S2", "time_value": 0.0, "x": 0.0, "y": 0.0, "z": 0.0},
            {"point_id": "q2", "strip_id": "S2", "time_value": 5.0, "x": 5.0, "y": 0.0, "z": 1.0},
            {"point_id": "q3", "strip_id": "S2", "time_value": 10.0, "x": 10.0, "y": 0.0, "z": 2.0},
            {"point_id": "q4", "strip_id": "S2", "time_value": 5.0, "x": 5.0, "y": 4.0, "z": 1.0},
            {"point_id": "q5", "strip_id": "S2", "time_value": 5.0, "x": 0.0, "y": 4.0, "z": 2.0},
        ],
        "controls": [
            {"control_id": "gcp-p1", "point_id": "p1", "x": 0.10, "y": 0.20, "z": 0.05, "weight": 10.0},
            {"control_id": "gcp-p3", "point_id": "p3", "x": 10.10, "y": 0.20, "z": 2.05, "weight": 10.0},
            {"control_id": "gcp-p4", "point_id": "p4", "x": 5.10, "y": 4.20, "z": 1.05, "weight": 10.0},
            {"control_id": "gcp-p5", "point_id": "p5", "x": 0.10, "y": 4.20, "z": 2.05, "weight": 10.0},
        ],
        "correspondences": [
            {"correspondence_id": "c-q1", "left_point_id": "p1", "right_point_id": "q1", "weight": 1.0, "group_id": "overlap-early", "metadata": {"intended_offset_m": [0.3, -0.2, 0.1]}},
            {"correspondence_id": "c-q2", "left_point_id": "p2", "right_point_id": "q2", "weight": 1.0, "group_id": "overlap-mid"},
            {"correspondence_id": "c-q3", "left_point_id": "p3", "right_point_id": "q3", "weight": 1.0, "group_id": "overlap-late", "metadata": {"drift_bias_m": [0.12, 0.0, 0.0]}},
            {"correspondence_id": "c-q4", "left_point_id": "p4", "right_point_id": "q4", "weight": 1.0, "group_id": "overlap-mid"},
            {"correspondence_id": "c-q5", "left_point_id": "p5", "right_point_id": "q5", "weight": 1.0, "group_id": "overlap-mid"},
            {"correspondence_id": "c-bad", "left_point_id": "p2", "right_point_id": "q4", "weight": 0.2, "group_id": "candidate-bad", "metadata": {"note": "deliberately poor candidate; useful for rejection demo"}},
        ],
    }


def seed_demo(connection) -> str | None:
    if many(connection, "SELECT 1 FROM datasets LIMIT 1"):
        return None
    result = import_dataset(connection, demo_request())
    return result["dataset_id"]

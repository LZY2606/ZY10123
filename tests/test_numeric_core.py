import math

from app.adjustment import ControlRecord, CorrespondenceRecord, PointRecord, canonical_segments, segment_for_point, solve_adjustment
from app.geometry import IDENTITY, apply_common_to_local, apply_local_to_common, exp_so3, segment_index


def test_transform_roundtrip_uses_documented_direction():
    rotation = exp_so3((0.03, -0.05, 0.08))
    translation = (1.2, -2.3, 0.4)
    point = (3.0, -4.0, 5.0)
    common = apply_local_to_common(point, rotation, translation)
    back = apply_common_to_local(common, rotation, translation)
    for expected, actual in zip(point, back):
        assert math.isclose(expected, actual, abs_tol=1e-12)


def test_boundary_time_belongs_only_to_following_segment():
    assert segment_index(9.999, [10.0]) == 0
    assert segment_index(10.0, [10.0]) == 1
    assert segment_index(10.001, [10.0]) == 1


def test_rank_deficiency_reports_null_directions_without_regularization():
    strips = [
        {"strip_id": "A", "time_start": 0.0, "time_end": 1.0},
        {"strip_id": "B", "time_start": 0.0, "time_end": 1.0},
    ]
    points = {
        "a": PointRecord("a", "A", 0.0, (0.0, 0.0, 0.0)),
        "b": PointRecord("b", "B", 0.0, (0.0, 0.0, 0.0)),
    }
    result = solve_adjustment(
        "strip_offset_v1", strips, points, [],
        [CorrespondenceRecord("c1", "a", "b")], {},
    )
    assert result["status"] == "degenerate"
    assert result["rank"] < result["degrees_of_freedom"]
    assert result["unconstrained_directions"]
    assert len(result["candidates"]) >= 3
    assert result["error_code"] == "rank_deficient"


def test_noncollinear_controls_make_rigid_model_full_rank():
    strips = [{"strip_id": "A", "time_start": 0.0, "time_end": 2.0}]
    points = {
        "p1": PointRecord("p1", "A", 0.0, (0.0, 0.0, 0.0)),
        "p2": PointRecord("p2", "A", 1.0, (1.0, 0.0, 0.0)),
        "p3": PointRecord("p3", "A", 2.0, (0.0, 1.0, 0.0)),
    }
    controls = [
        ControlRecord("g1", "p1", (0.1, 0.0, 0.0)),
        ControlRecord("g2", "p2", (1.1, 0.0, 0.0)),
        ControlRecord("g3", "p3", (0.1, 1.0, 0.0)),
    ]
    result = solve_adjustment("rigid_v1", strips, points, controls, [], {})
    assert result["status"] == "ok"
    assert result["rank"] == result["degrees_of_freedom"] == 6
    for control_id in ("g1", "g2", "g3"):
        assert result["residuals"][control_id]["norm"] < 1e-9


def test_exact_boundary_point_uses_second_segment_split_strictly_inside():
    strip = {"strip_id": "A", "time_start": 0.0, "time_end": 10.0, "split_time": 5.0}
    point = PointRecord("p", "A", 5.0, (0, 0, 0))
    index, _ = segment_for_point(point, [strip])
    assert index == 1


def test_linear_drift_parameters_are_recovered_deterministically():
    strips = [{"strip_id": "A", "time_start": 0.0, "time_end": 10.0}]
    points = {
        "p0": PointRecord("p0", "A", 0.0, (0.0, 0.0, 0.0)),
        "p1": PointRecord("p1", "A", 10.0, (1.0, 0.0, 0.0)),
        "p2": PointRecord("p2", "A", 10.0, (0.0, 1.0, 0.0)),
        "p3": PointRecord("p3", "A", 10.0, (0.0, 0.0, 1.0)),
        "p4": PointRecord("p4", "A", 5.0, (1.0, 1.0, 0.0)),
    }
    translation = (0.2, -0.1, 0.05)
    slope = (0.3, 0.2, 0.1)

    def target(point):
        fraction = point.time_value / 10.0
        return tuple(point.local[axis] + translation[axis] + slope[axis] * fraction for axis in range(3))

    controls = [ControlRecord(f"g{index}", point.point_id, target(point)) for index, point in enumerate(points.values())]
    result = solve_adjustment("drift_v1", strips, points, controls, [], {})
    assert result["status"] == "ok"
    state = result["transform"]["segments"]["A#segment0"]
    for expected, actual in zip(translation, state["t"]):
        assert math.isclose(expected, actual, abs_tol=1e-12)
    for expected, actual in zip(slope, state["a"]):
        assert math.isclose(expected, actual, abs_tol=1e-12)


def test_locked_control_is_an_exact_equality_constraint():
    strips = [{"strip_id": "A", "time_start": 0.0, "time_end": 2.0}]
    points = {
        "p1": PointRecord("p1", "A", 0.0, (0.0, 0.0, 0.0)),
        "p2": PointRecord("p2", "A", 1.0, (1.0, 0.0, 0.0)),
        "p3": PointRecord("p3", "A", 2.0, (0.0, 1.0, 0.0)),
    }
    controls = [
        ControlRecord("g1", "p1", (0.1, 0.0, 0.0)),
        ControlRecord("g2", "p2", (1.1, 0.0, 0.0)),
        ControlRecord("g3", "p3", (0.1, 1.0, 0.0)),
    ]
    result = solve_adjustment("rigid_v1", strips, points, controls, [], {"locked_control_ids": ["g1"]})
    assert result["status"] == "ok"
    assert result["residuals"]["g1"]["norm"] == 0.0
    assert result["locked_control_ids"] == ["g1"]


def test_small_rotation_is_recovered_with_documented_left_composition():
    from app.geometry import exp_so3

    strips = [{"strip_id": "A", "time_start": 0.0, "time_end": 2.0}]
    local = {
        "p0": (0.0, 0.0, 0.0),
        "p1": (1.0, 0.0, 0.0),
        "p2": (0.0, 1.0, 0.0),
        "p3": (0.0, 0.0, 1.0),
    }
    omega = (0.02, -0.03, 0.05)
    translation = (0.2, -0.1, 0.05)
    rotation = exp_so3(omega)

    def target(coordinates):
        rotated = tuple(sum(rotation[i][j] * coordinates[j] for j in range(3)) for i in range(3))
        return tuple(rotated[i] + translation[i] for i in range(3))

    points = {point_id: PointRecord(point_id, "A", 1.0, coordinates) for point_id, coordinates in local.items()}
    controls = [ControlRecord(f"g{point_id[1:]}", point_id, target(coordinates)) for point_id, coordinates in local.items()]
    result = solve_adjustment("rigid_v1", strips, points, controls, [], {})
    assert result["status"] == "ok"
    for expected, actual in zip(omega, result["transform"]["omega"]):
        assert math.isclose(expected, actual, abs_tol=1e-12)
    for expected, actual in zip(translation, result["transform"]["segments"]["__common__#segment0"]["t"]):
        assert math.isclose(expected, actual, abs_tol=1e-12)


def test_correspondence_jacobian_direction_matches_right_minus_left_residual():
    strips = [
        {"strip_id": "A", "time_start": 0.0, "time_end": 1.0},
        {"strip_id": "B", "time_start": 0.0, "time_end": 1.0},
    ]
    points = {
        "a0": PointRecord("a0", "A", 0.0, (0.0, 0.0, 0.0)),
        "a1": PointRecord("a1", "A", 1.0, (1.0, 0.0, 0.0)),
        "a2": PointRecord("a2", "A", 1.0, (0.0, 1.0, 0.0)),
        "b0": PointRecord("b0", "B", 0.0, (0.0, 0.0, 0.0)),
        "b1": PointRecord("b1", "B", 1.0, (1.0, 0.0, 0.0)),
        "b2": PointRecord("b2", "B", 1.0, (0.0, 1.0, 0.0)),
    }
    shift = (0.1, -0.2, 0.05)
    controls = [
        ControlRecord("g0", "a0", shift),
        ControlRecord("g1", "a1", (1.0 + shift[0], shift[1], shift[2])),
        ControlRecord("g2", "a2", (shift[0], 1.0 + shift[1], shift[2])),
    ]
    correspondences = [
        CorrespondenceRecord("c0", "a0", "b0"),
        CorrespondenceRecord("c1", "a1", "b1"),
        CorrespondenceRecord("c2", "a2", "b2"),
    ]
    result = solve_adjustment("strip_offset_v1", strips, points, controls, correspondences, {})
    assert result["status"] == "ok"
    for correspondence_id in ("c0", "c1", "c2"):
        assert result["residuals"][correspondence_id]["norm"] < 1e-10
    b_state = result["transform"]["segments"]["B#segment0"]["t"]
    for expected, actual in zip(shift, b_state):
        assert math.isclose(expected, actual, abs_tol=1e-12)


def test_split_strip_uses_separate_translations_and_boundary_goes_to_second_segment():
    strips = [{"strip_id": "A", "time_start": 0.0, "time_end": 10.0, "split_time": 5.0}]
    points = {
        "before": PointRecord("before", "A", 0.0, (0.0, 0.0, 0.0)),
        "after": PointRecord("after", "A", 10.0, (0.0, 1.0, 0.0)),
        "boundary": PointRecord("boundary", "A", 5.0, (1.0, 0.0, 0.0)),
        "side": PointRecord("side", "A", 2.0, (0.0, 0.0, 1.0)),
    }
    controls = [
        ControlRecord("g-before", "before", (1.0, 0.0, 0.0)),
        ControlRecord("g-after", "after", (1.0, 1.0, 1.0)),
        ControlRecord("g-boundary", "boundary", (2.0, 0.0, 1.0)),
        ControlRecord("g-side", "side", (1.0, 0.0, 1.0)),
    ]
    result = solve_adjustment("strip_offset_v1", strips, points, controls, [], {})
    assert result["status"] == "ok"
    first_translation = result["transform"]["segments"]["A#segment0"]["t"]
    second_translation = result["transform"]["segments"]["A#segment1"]["t"]
    for expected, actual in zip((1.0, 0.0, 0.0), first_translation):
        assert math.isclose(expected, actual, abs_tol=1e-12)
    for expected, actual in zip((1.0, 0.0, 1.0), second_translation):
        assert math.isclose(expected, actual, abs_tol=1e-12)
    assert result["residuals"]["g-boundary"]["norm"] < 1e-12

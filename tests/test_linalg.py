from app.linalg import rank_revealing_least_squares


def test_empty_design_reports_every_parameter_as_unconstrained():
    result = rank_revealing_least_squares(design=[], response=[], cols=3)
    assert result["status"] == "ok"
    assert result["rank"] == 0
    assert result["degrees_of_freedom"] == 3
    assert len(result["null_basis"]) == 3
    assert result["solution"] == [0.0, 0.0, 0.0]

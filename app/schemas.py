from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class ImportRequest(BaseModel):
    job_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    common_crs: str = Field(min_length=1)
    common_length_unit: Literal["m", "ft", "survey_ft"]
    metadata: dict[str, Any] = Field(default_factory=dict)
    strips: list[dict[str, Any]]
    points: list[dict[str, Any]]
    controls: list[dict[str, Any]] = Field(default_factory=list)
    correspondences: list[dict[str, Any]] = Field(default_factory=list)


class SolveRequest(BaseModel):
    job_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    model: Literal["rigid_v1", "strip_offset_v1", "drift_v1"]
    parent_scheme_id: str | None = None
    disabled_control_ids: list[str] = Field(default_factory=list)
    locked_control_ids: list[str] = Field(default_factory=list)
    disabled_correspondence_ids: list[str] = Field(default_factory=list)
    disabled_correspondence_group_ids: list[str] = Field(default_factory=list)
    weights: dict[str, float] = Field(default_factory=dict)
    segment_splits: dict[str, float] = Field(default_factory=dict)
    residual_limit_m: float | None = None
    max_iterations: int = 20
    step_tolerance: float = 1e-11


class PublishRequest(BaseModel):
    job_id: str = Field(min_length=1)
    scheme_id: str = Field(min_length=1)

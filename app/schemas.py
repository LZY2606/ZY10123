from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PointIn(BaseModel):
    id: int
    x: float
    y: float
    z: float
    timestamp_s: Optional[float] = None


class StripIn(BaseModel):
    strip_id: str
    points: List[PointIn]


class GcpIn(BaseModel):
    point_id: int
    code: str
    control_x: float
    control_y: float
    control_z: float
    control_unit: str = "m"
    control_crs: Optional[str] = None


class CorrIn(BaseModel):
    point_a_id: int
    point_b_id: int


class PointsetIn(BaseModel):
    name: str = "pointset"
    unit: str = "m"
    crs: Optional[str] = None
    source_description: Optional[str] = None
    strips: List[StripIn]
    gcps: List[GcpIn] = []
    correspondences: List[CorrIn] = []
    simulate_crash: bool = False


class PlanIn(BaseModel):
    name: str
    model_version: str = "rigid6-v1"
    drift_breaks: Dict[str, List[float]] = {}
    weights: Dict[str, float] = {}
    lock_factor: Optional[float] = None
    outlier_sigma: Optional[float] = None
    reject_outliers: Optional[bool] = None
    disabled_gcp_ids: List[int] = []
    locked_gcp_ids: List[int] = []
    disabled_corr_ids: List[int] = []
    disabled_corr_groups: List[str] = []


class ForkIn(BaseModel):
    name: str
    changes: Dict[str, Any] = Field(default_factory=dict)


class JobIn(BaseModel):
    simulate_crash: bool = False

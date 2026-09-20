"""FastAPI application: LiDAR strip alignment workbench."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import service
from .db import connect, init_db
from .schemas import ForkIn, JobIn, PlanIn, PointsetIn
from .fingerprint import RULE_VERSION

DB_PATH = os.environ.get("LIDAR_ALIGN_DB", "lidar_align.db")

app = FastAPI(title="LiDAR Strip Alignment Workbench", version="1.0.0")


@app.on_event("startup")
def startup() -> None:
    conn = connect(DB_PATH)
    init_db(conn)
    app.state.db = conn


@app.exception_handler(service.ServiceError)
def service_error_handler(request: Request, exc: service.ServiceError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "rule_version": RULE_VERSION}


@app.get("/api/demo")
def demo_payload_endpoint() -> dict:
    from .demo import demo_payload
    return demo_payload()


@app.post("/api/pointsets")
def create_pointset(payload: PointsetIn) -> dict:
    try:
        return service.import_pointset(app.state.db, payload.model_dump())
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/pointsets/{pointset_id}")
def read_pointset(pointset_id: int) -> dict:
    return service.get_pointset(app.state.db, pointset_id)


@app.get("/api/pointsets/{pointset_id}/data")
def pointset_data(pointset_id: int) -> dict:
    conn = app.state.db
    pointset = service.get_pointset(conn, pointset_id)
    strips = service.list_strip_points(conn, pointset_id)
    gcps = [
        dict(r) for r in conn.execute(
            "SELECT * FROM gcps WHERE pointset_id=? ORDER BY id", (pointset_id,)
        ).fetchall()
    ]
    corrs = [
        dict(r) for r in conn.execute(
            "SELECT * FROM correspondences WHERE pointset_id=? ORDER BY id",
            (pointset_id,),
        ).fetchall()
    ]
    return {"pointset": pointset, "strips": list(strips.values()),
            "gcps": gcps, "correspondences": corrs}


@app.post("/api/pointsets/{pointset_id}/plans")
def create_plan(pointset_id: int, payload: PlanIn) -> dict:
    plan_id = service.create_plan(
        app.state.db, pointset_id, payload.name, payload.model_dump(exclude={"name"})
    )
    return service.get_plan(app.state.db, plan_id)


@app.get("/api/plans/{plan_id}")
def read_plan(plan_id: int) -> dict:
    return service.get_plan(app.state.db, plan_id)


@app.post("/api/plans/{plan_id}/fork")
def fork_plan(plan_id: int, payload: ForkIn) -> dict:
    new_id = service.fork_plan(app.state.db, plan_id, payload.name, payload.changes)
    return service.get_plan(app.state.db, new_id)


@app.post("/api/plans/{plan_id}/publish")
def publish_plan(plan_id: int) -> dict:
    service.publish_plan(app.state.db, plan_id)
    return service.get_plan(app.state.db, plan_id)


@app.post("/api/plans/{plan_id}/jobs")
def run_plan_job(plan_id: int, payload: Optional[JobIn] = None) -> dict:
    simulate_crash = bool(payload.simulate_crash) if payload is not None else False
    try:
        job, replayed = service.run_job(
            app.state.db, plan_id, simulate_crash=simulate_crash
        )
    except RuntimeError as exc:
        # The simulated failure is recorded in-db; surface it to the caller.
        raise HTTPException(status_code=500, detail=str(exc))
    return {"replayed": replayed, "job": job}


@app.get("/api/plans/{plan_id}/latest")
def plan_latest(plan_id: int) -> dict:
    """Always-200 probe: returns the latest succeeded result or null.

    Lets the UI check solved state without a 409 appearing as a failed
    resource load in the browser console.
    """
    service.get_plan(app.state.db, plan_id)
    return {"result": service.latest_result(app.state.db, plan_id)}


@app.get("/api/plans/{plan_id}/corrected")
def plan_corrected(plan_id: int) -> dict:
    return service.corrected_points(app.state.db, plan_id)


@app.get("/api/pointsets/{pointset_id}/plans")
def pointset_plans(pointset_id: int) -> dict:
    return {"plans": service.list_plans(app.state.db, pointset_id)}


@app.get("/api/plans/{plan_a}/compare/{plan_b}")
def compare_plans(plan_a: int, plan_b: int) -> dict:
    conn = app.state.db
    pa = service.get_plan(conn, plan_a)
    pb = service.get_plan(conn, plan_b)
    if pa["pointset_id"] != pb["pointset_id"]:
        raise HTTPException(status_code=400, detail="plans belong to different pointsets")
    ca = service.corrected_points(conn, plan_a)
    cb = service.corrected_points(conn, plan_b)
    ra = service.latest_result(conn, plan_a)
    rb = service.latest_result(conn, plan_b)
    return {
        "plan_a": {"plan": pa, "points": ca, "result": ra["result"] if ra else None},
        "plan_b": {"plan": pb, "points": cb, "result": rb["result"] if rb else None},
    }


@app.get("/api/audit")
def audit(limit: int = Query(200, ge=1, le=1000)) -> dict:
    return {"events": service.audit_log(app.state.db, limit)}


static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/favicon.ico")
def favicon() -> Response:
    # Empty 204 (no body) so the browser does not log a missing favicon.
    return Response(status_code=204)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(static_dir / "index.html")

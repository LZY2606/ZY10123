from __future__ import annotations

import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import services
from .schemas import ImportRequest, PublishRequest, SolveRequest
from .database import SCHEMA, connect
from .demo import seed_demo

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app):
    connection = get_connection()
    try:
        with connection:
            connection.executescript(SCHEMA)
            if os.environ.get("SEED_DEMO", "1") != "0":
                seed_demo(connection)
        yield
    finally:
        connection.close()


app = FastAPI(title="Pairwise LiDAR Strip Adjustment", version="1.0.0", lifespan=lifespan)


def get_connection():
    return connect()


@app.exception_handler(services.ServiceError)
def service_error_handler(request: Request, exc: services.ServiceError):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=exc.status_code, content={"error_code": exc.error_code, "message": str(exc), "details": exc.details})


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/datasets")
def datasets():
    with get_connection() as connection:
        return {"datasets": services.list_datasets(connection)}


@app.post("/api/datasets/import")
def import_dataset(payload: ImportRequest):
    with get_connection() as connection:
        return services.import_dataset(connection, payload.model_dump())


@app.get("/api/datasets/{dataset_id}")
def dataset_detail(dataset_id: str):
    with get_connection() as connection:
        return services.get_dataset_detail(connection, dataset_id)


@app.get("/api/datasets/{dataset_id}/schemes")
def dataset_schemes(dataset_id: str):
    with get_connection() as connection:
        return {"schemes": services.list_schemes(connection, dataset_id)}


@app.post("/api/schemes/solve")
def solve_scheme(payload: SolveRequest):
    with get_connection() as connection:
        return services.solve_scheme(connection, payload.model_dump())


@app.post("/api/schemes/publish")
def publish_scheme(payload: PublishRequest):
    with get_connection() as connection:
        return services.publish_scheme(connection, payload.model_dump())


@app.get("/api/schemes/{scheme_id}")
def scheme_detail(scheme_id: str):
    with get_connection() as connection:
        return services.get_scheme_detail(connection, scheme_id)


@app.get("/api/compare/{left_id}/{right_id}")
def compare(left_id: str, right_id: str):
    with get_connection() as connection:
        return services.compare_schemes(connection, left_id, right_id)


@app.get("/api/jobs/{job_id}")
def job_detail(job_id: str):
    with get_connection() as connection:
        return services.get_job_detail(connection, job_id)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

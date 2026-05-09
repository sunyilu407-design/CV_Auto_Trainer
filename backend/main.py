import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from routers import tasks, vlm, settings, training, files, auth, algorithm, models as models_router, reasoning, negotiate
from models.database import SessionLocal
from routers.auth import seed_admin_user


@asynccontextmanager
async def lifespan(app: FastAPI):
    db = SessionLocal()
    try:
        seed_admin_user(db)
    finally:
        db.close()
    yield


app = FastAPI(title="CV Auto Trainer Backend", version="1.0.0", lifespan=lifespan)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORS_ORIGINS = [
    item.strip()
    for item in os.getenv(
        "CV_AUTO_TRAINER_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if item.strip()
]


def resolve_frontend_dist(
    frontend_dist_env: str | None = None,
    project_root: Path | None = None,
    cwd: Path | None = None,
) -> Path:
    env_value = (frontend_dist_env if frontend_dist_env is not None else os.getenv("CV_AUTO_TRAINER_FRONTEND_DIST", "")).strip()
    root = project_root or PROJECT_ROOT
    current_dir = cwd or Path.cwd()
    candidates: list[Path] = []

    if env_value:
        env_path = Path(env_value).expanduser()
        if env_path.is_absolute():
            candidates.append(env_path)
            if env_path.name != "dist":
                candidates.append(env_path / "dist")
        else:
            cwd_candidate = current_dir / env_path
            root_candidate = root / env_path
            candidates.extend([cwd_candidate, root_candidate])
            if env_path.name != "dist":
                candidates.extend([cwd_candidate / "dist", root_candidate / "dist"])

    candidates.append(root / "frontend" / "dist")

    seen: set[Path] = set()
    normalized_candidates: list[Path] = []
    for candidate in candidates:
        normalized = candidate.resolve(strict=False)
        if normalized in seen:
            continue
        seen.add(normalized)
        normalized_candidates.append(normalized)

    for candidate in normalized_candidates:
        if (candidate / "index.html").exists():
            return candidate

    return normalized_candidates[0]


FRONTEND_DIST = resolve_frontend_dist()

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tasks.router)
app.include_router(vlm.router)
app.include_router(algorithm.router)
app.include_router(settings.router)
app.include_router(training.router)
app.include_router(files.router)
app.include_router(auth.router)
app.include_router(models_router.router)
app.include_router(reasoning.router)
app.include_router(negotiate.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}


dist_path = FRONTEND_DIST
assets_path = dist_path / "assets"
if assets_path.exists():
    app.mount("/assets", StaticFiles(directory=str(assets_path)), name="frontend-assets")


@app.get("/", include_in_schema=False)
@app.get("/{full_path:path}", include_in_schema=False)
def serve_frontend(full_path: str = ""):
    if full_path.startswith(("api/", "docs", "openapi.json", "redoc", "assets/")):
        return {"detail": "Not Found"}

    index_path = dist_path / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"detail": "Frontend dist not found"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

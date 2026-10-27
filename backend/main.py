from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.api.routes import router
from backend.database.models import init_db
from backend.paths import FRONTEND_DIR


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── startup ──
    init_db()
    from backend.core.ai_processor import detector, camera
    try:
        detector.load_known_faces()
    except Exception as e:
        print(f"[WARN] Could not load faces on startup: {e}")
    # Camera opens in the background; the app stays usable even without one.
    camera.start()
    print("[INFO] AIFACE is ready.")
    yield
    # ── shutdown ──
    try:
        camera.stop()
    except Exception:
        pass


app = FastAPI(title="AIFACE Premium Backend", lifespan=lifespan)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def no_cache_frontend(request, call_next):
    # Without this the browser kept serving stale index.html/style.css after
    # an update (heuristic caching), so fixes didn't show up until a hard reload.
    response = await call_next(request)
    if not request.url.path.startswith("/api"):
        response.headers["Cache-Control"] = "no-cache"
    return response


# Include API routes
app.include_router(router, prefix="/api")

# Mount frontend static files (must be last so /api wins)
frontend_path = FRONTEND_DIR
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000)

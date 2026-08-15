from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import Settings, get_settings
from app.pipeline import AnalysisPipeline
from app.schemas import AnalyzeRequest, AnalyzeResponse


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    pipeline = AnalysisPipeline(settings)
    app = FastAPI(title=settings.app_name, version="1.0.0")
    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    def dashboard() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "fusion_model": "quality-aware state-space + RTS smoothing"}

    @app.post("/analyze", response_model=AnalyzeResponse)
    def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
        try:
            return pipeline.analyze(request)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    return app


app = create_app()

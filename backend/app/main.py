"""Веб-адаптер общего ядра: чат, карточки источников и раздача фронтенда."""
import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from assistant_core.contracts import STATUS_ERROR, TechnicalError
from backend.app.api import chat, sources

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("web")
ROOT = Path(__file__).resolve().parents[2]

app = FastAPI(title="Юридический ассистент ПСБ")
app.include_router(chat.router)
app.include_router(sources.router)


@app.exception_handler(TechnicalError)
async def technical_error(request: Request, exc: TechnicalError) -> JSONResponse:
    logger.error("Техническая ошибка на %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=503,
        content={"status": STATUS_ERROR, "text": "Техническая ошибка. Повторите запрос позже."},
    )


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


if (ROOT / "frontend").is_dir():
    app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
"""Карточка источника и исходный текст по ID. Модель здесь не вызывается."""
from dataclasses import asdict
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException

from assistant_core.assistant_service import Assistant, get_assistant
from assistant_core.citations import source_card
from assistant_core.contracts import SourceCard, TechnicalError

router = APIRouter()


def card_to_json(card: SourceCard) -> dict:
    data = asdict(card)
    data["source_url"] = (
        f"/api/documents/{quote(card.document_id, safe='')}"
        f"?line_start={card.line_start}&line_end={card.line_end}"
    )
    return data


@router.get("/api/sources/{fragment_id}")
def get_source(fragment_id: str, assistant: Assistant = Depends(get_assistant)) -> dict:
    try:
        card = source_card(assistant.store, fragment_id)
    except Exception as exc:
        raise TechnicalError(f"каталог недоступен: {exc}") from exc
    if card is None:
        raise HTTPException(status_code=404, detail="Фрагмент не найден")
    return card_to_json(card)


@router.get("/api/documents/{document_id}")
def get_document(
    document_id: str,
    line_start: int = 0,
    line_end: int = 0,
    assistant: Assistant = Depends(get_assistant),
) -> dict:
    try:
        text = assistant.store.get_document_text(document_id)
    except Exception as exc:
        raise TechnicalError(f"каталог недоступен: {exc}") from exc
    if text is None:
        raise HTTPException(status_code=404, detail="Документ не найден")
    return {
        "document_id": document_id,
        "text": text,
        "line_start": line_start,
        "line_end": line_end,
    }
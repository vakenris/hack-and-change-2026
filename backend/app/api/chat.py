"""POST /api/chat: принимает диалог, вызывает ядро, возвращает статус, текст и ссылки."""
from dataclasses import asdict
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from assistant_core.assistant_service import Assistant, get_assistant
from assistant_core.contracts import Answer, Message
from backend.app.api.sources import card_to_json

router = APIRouter()


class MessageIn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=20000)


class ChatRequest(BaseModel):
    messages: list[MessageIn] = Field(min_length=1, max_length=50)


def answer_to_json(answer: Answer) -> dict:
    return {
        "status": answer.status,
        "text": answer.text,
        "claims": [asdict(claim) for claim in answer.claims],
        "sources": [card_to_json(card) for card in answer.sources],
        "missing_fact": answer.missing_fact,
        "refusal_reason": answer.refusal_reason,
    }


@router.post("/api/chat")
def chat(request: ChatRequest, assistant: Assistant = Depends(get_assistant)) -> dict:
    messages = [Message(item.role, item.content) for item in request.messages]
    try:
        answer = assistant.answer(messages)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return answer_to_json(answer)
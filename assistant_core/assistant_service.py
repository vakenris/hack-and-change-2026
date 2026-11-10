"""Общее ядро: один вход для веб-чата и solution.py."""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from . import citations
from .contracts import (
    STATUS_CLARIFY,
    STATUS_FULL,
    STATUS_PARTIAL,
    STATUS_REFUSAL,
    Answer,
    Deps,
    Message,
    TechnicalError,
)

logger = logging.getLogger("assistant")
ROOT = Path(__file__).resolve().parent.parent
NO_GROUNDS = "В выданной базе не найдено положений, подтверждающих ответ на этот вопрос."
KNOWN_STATUSES = (STATUS_FULL, STATUS_PARTIAL, STATUS_CLARIFY, STATUS_REFUSAL)


class Assistant:
    """Не хранит состояние между вызовами: каждый диалог обрабатывается независимо."""

    def __init__(self, deps: Deps, search_limit: int = 12) -> None:
        self.deps = deps
        self.store = deps.store
        self.search_limit = search_limit

    def answer(self, messages: list[Message]) -> Answer:
        cleaned = [
            Message(m.role, m.content.strip()) for m in messages if m.content and m.content.strip()
        ]
        if not cleaned or cleaned[-1].role != "user":
            raise ValueError("Последняя реплика должна быть вопросом пользователя")
        try:
            return self._run(cleaned)
        except TechnicalError:
            raise
        except Exception as exc:
            logger.exception("Сбой при обработке вопроса")
            raise TechnicalError(f"{type(exc).__name__}: {exc}") from exc

    def _run(self, messages: list[Message]) -> Answer:
        deps = self.deps
        facts = deps.extract_facts(messages)
        candidates = deps.search(facts, self.search_limit)
        selection = deps.select(candidates, facts)
        passed = {fragment.fragment_id: fragment for fragment in selection.applicable}

        draft = deps.draft(messages, facts, selection.applicable)
        checked, rejected = citations.filter_claims(draft.claims, passed, deps.store)
        if rejected:
            logger.warning("Отброшены ссылки вне каталога или вне переданных фрагментов: %s", rejected)

        verified = deps.verify(checked, passed) if checked else []
        verified, _ = citations.filter_claims(verified, passed, deps.store)

        decision = deps.decide(facts, selection, draft, verified)
        status = decision.status
        text = (decision.text or "").strip()
        reason = decision.refusal_reason
        if status not in KNOWN_STATUSES:
            raise TechnicalError(f"Неизвестный статус политики ответа: {status}")
        if status in (STATUS_FULL, STATUS_PARTIAL) and not verified:
            status, text = STATUS_REFUSAL, NO_GROUNDS
            reason = reason or "нет подтверждённых оснований"
        if not text:
            raise TechnicalError("Политика ответа вернула пустой текст")

        return Answer(
            status=status,
            text=text,
            claims=verified,
            sources=citations.build_sources(verified, deps.store),
            missing_fact=decision.missing_fact,
            refusal_reason=reason,
        )


def default_db_path() -> str:
    return os.environ.get("KNOWLEDGE_DB") or str(ROOT / "data" / "processed" / "knowledge.sqlite")


def build_assistant(db_path: Optional[str] = None) -> Assistant:
    if os.environ.get("ASSISTANT_STUB") == "1":
        from .stubs import build_stub_deps

        return Assistant(build_stub_deps())

    path = db_path or default_db_path()
    if not Path(path).is_file():
        raise TechnicalError(f"Не найдена база знаний: {path}")
    try:
        from . import answer_policy, knowledge_store, llm_client, retrieval, versions

        store = knowledge_store.open_store(path)
    except Exception as exc:
        raise TechnicalError(f"Ядро не запущено: {type(exc).__name__}: {exc}") from exc

    return Assistant(
        Deps(
            store=store,
            search=lambda facts, limit: retrieval.search(store, facts, limit),
            select=lambda candidates, facts: versions.select(store, candidates, facts),
            extract_facts=llm_client.extract_facts,
            draft=llm_client.draft_answer,
            verify=llm_client.verify_claims,
            decide=answer_policy.decide,
        )
    )


@lru_cache(maxsize=1)
def get_assistant() -> Assistant:
    """Один экземпляр на процесс веб-сервера."""
    return build_assistant()
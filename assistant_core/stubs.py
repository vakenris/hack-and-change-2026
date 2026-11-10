"""Заглушки ядра для фронтенда и тестов стыков. Тексты пунктов условные."""
from __future__ import annotations

import re
from datetime import date
from typing import Optional

from .contracts import (
    STATUS_CLARIFY,
    STATUS_FULL,
    STATUS_PARTIAL,
    STATUS_REFUSAL,
    Claim,
    Decision,
    Deps,
    Draft,
    EventDate,
    Fragment,
    Message,
    QueryFacts,
    Selection,
    TechnicalError,
)

_TITLE = "Договор поставки № П-17/2024"
_DOC_LINES = [
    _TITLE,
    "3.2. Условный текст: товар доставляется на склад № 1.",
    "3.2. Условный текст: товар доставляется на склад № 2.",
]
_FRAGMENTS = {
    "F-17": Fragment(
        "F-17", "D-P17", _TITLE, "R-0", "исходная редакция", "п. 3.2",
        _DOC_LINES[1], "data/raw/stub.txt", 2, 2,
    ),
    "F-42": Fragment(
        "F-42", "D-P17", _TITLE, "R-1", "в редакции допсоглашения № 1", "п. 3.2",
        _DOC_LINES[2], "data/raw/stub.txt", 3, 3,
    ),
}
_BORDER = date(2024, 10, 1)
_DATE_RE = re.compile(r"\b(\d{2})\.(\d{2})\.(\d{4})\b")


class StubStore:
    def get_fragment(self, fragment_id: str) -> Optional[Fragment]:
        return _FRAGMENTS.get(fragment_id)

    def get_document_text(self, document_id: str) -> Optional[str]:
        return "\n".join(_DOC_LINES) if document_id == "D-P17" else None


def _extract_facts(messages: list[Message]) -> QueryFacts:
    user_texts = [m.content for m in messages if m.role == "user"]
    dates = []
    for text in user_texts:
        for day, month, year in _DATE_RE.findall(text):
            dates.append(EventDate("spec_signed", date(int(year), int(month), int(day))))
    return QueryFacts(question=user_texts[-1], search_queries=user_texts, dates=dates)


def _search(facts: QueryFacts, limit: int) -> list[Fragment]:
    text = " ".join(facts.search_queries).lower()
    if "сбой" in text:
        raise TechnicalError("индекс недоступен (заглушка)")
    if "адрес" in text or "п-17" in text:
        return list(_FRAGMENTS.values())[:limit]
    return []


def _select(candidates: list[Fragment], facts: QueryFacts) -> Selection:
    if not candidates:
        return Selection()
    if not facts.dates:
        return Selection(
            gaps=["не указана дата подписания спецификации"],
            missing_fact="дата подписания спецификации",
        )
    chosen = "F-42" if facts.dates[-1].value > _BORDER else "F-17"
    gaps = ["в базе нет положения о штрафе"] if "штраф" in facts.question.lower() else []
    return Selection(applicable=[_FRAGMENTS[chosen]], gaps=gaps)


def _draft(messages: list[Message], facts: QueryFacts, fragments: list[Fragment]) -> Draft:
    if "обойти" in facts.question.lower():
        return Draft(refusal_reason="просьба о содействии в обходе требований")
    claims = [
        # F-999 добавлен намеренно: проверка ссылок должна его отбросить
        Claim(f"Применяется {f.clause} ({f.revision_label}). {f.text}", [f.fragment_id, "F-999"])
        for f in fragments
    ]
    return Draft(claims=claims)


def _verify(claims: list[Claim], fragments: dict[str, Fragment]) -> list[Claim]:
    return claims


def _decide(
    facts: QueryFacts, selection: Selection, draft: Draft, verified: list[Claim]
) -> Decision:
    if draft.refusal_reason:
        return Decision(
            STATUS_REFUSAL,
            "Не могу помочь с этим запросом: " + draft.refusal_reason + ".",
            refusal_reason=draft.refusal_reason,
        )
    if not verified and selection.missing_fact:
        return Decision(
            STATUS_CLARIFY,
            "Уточните, пожалуйста: " + selection.missing_fact + ".",
            missing_fact=selection.missing_fact,
        )
    if not verified:
        return Decision(
            STATUS_REFUSAL,
            "В базе нет положений для ответа на этот вопрос.",
            refusal_reason="нет оснований в базе",
        )
    text = " ".join(claim.text for claim in verified)
    if selection.gaps:
        return Decision(STATUS_PARTIAL, text + " Не подтверждено: " + "; ".join(selection.gaps) + ".")
    return Decision(STATUS_FULL, text)


def build_stub_deps() -> Deps:
    return Deps(
        store=StubStore(),
        search=_search,
        select=_select,
        extract_facts=_extract_facts,
        draft=_draft,
        verify=_verify,
        decide=_decide,
    )
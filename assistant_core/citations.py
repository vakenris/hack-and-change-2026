"""Проверка ссылок по каталогу и карточки источников."""
from __future__ import annotations

from typing import Optional

from .contracts import Claim, Fragment, KnowledgeStore, SourceCard


def filter_claims(
    claims: list[Claim], passed: dict[str, Fragment], store: KnowledgeStore
) -> tuple[list[Claim], list[str]]:
    """Оставляет только ID, которые есть в каталоге и были переданы модели."""
    kept: list[Claim] = []
    rejected: list[str] = []
    for claim in claims:
        valid: list[str] = []
        for fragment_id in claim.fragment_ids:
            if fragment_id in passed and store.get_fragment(fragment_id) is not None:
                if fragment_id not in valid:
                    valid.append(fragment_id)
            else:
                rejected.append(fragment_id)
        if valid and claim.text.strip():
            kept.append(Claim(text=claim.text.strip(), fragment_ids=valid))
    return kept, rejected


def source_card(store: KnowledgeStore, fragment_id: str) -> Optional[SourceCard]:
    """Цитата и координаты берутся из каталога, а не из ответа модели."""
    fragment = store.get_fragment(fragment_id)
    if fragment is None:
        return None
    return SourceCard(
        fragment_id=fragment.fragment_id,
        document_id=fragment.document_id,
        document_title=fragment.document_title,
        revision_label=fragment.revision_label,
        clause=fragment.clause,
        quote=fragment.text,
        line_start=fragment.line_start,
        line_end=fragment.line_end,
    )


def build_sources(claims: list[Claim], store: KnowledgeStore) -> list[SourceCard]:
    cards: list[SourceCard] = []
    seen: set[str] = set()
    for claim in claims:
        for fragment_id in claim.fragment_ids:
            if fragment_id in seen:
                continue
            card = source_card(store, fragment_id)
            if card is not None:
                seen.add(fragment_id)
                cards.append(card)
    return cards
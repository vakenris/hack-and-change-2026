"""Общие типы ядра. Меняются только по согласованию со всей командой."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Optional, Protocol

STATUS_FULL = "full"
STATUS_PARTIAL = "partial"
STATUS_CLARIFY = "clarify"
STATUS_REFUSAL = "refusal"
STATUS_ERROR = "error"  # только в ответе веб-API при техническом сбое


class TechnicalError(RuntimeError):
    """Сбой модели, базы или индекса. Не является юридическим отказом."""


@dataclass(frozen=True)
class Message:
    role: str  # "user" или "assistant"
    content: str


@dataclass(frozen=True)
class EventDate:
    event_type: str  # например "spec_signed", "delivery", "deal", "unknown"
    value: date


@dataclass
class QueryFacts:
    question: str  # последняя реплика пользователя
    search_queries: list[str] = field(default_factory=list)
    document_refs: list[str] = field(default_factory=list)  # номера договоров, статей, пунктов
    dates: list[EventDate] = field(default_factory=list)


@dataclass(frozen=True)
class Fragment:
    fragment_id: str  # стабильный ID: латиница, цифры, дефис, подчёркивание
    document_id: str
    document_title: str
    revision_id: str
    revision_label: str
    clause: str
    text: str  # дословный текст из исходника
    source_path: str
    line_start: int  # номера строк в TXT с единицы, включительно
    line_end: int


@dataclass
class Selection:
    applicable: list[Fragment] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)  # причины неопределённости
    missing_fact: Optional[str] = None


@dataclass
class Claim:
    text: str
    fragment_ids: list[str] = field(default_factory=list)


@dataclass
class Draft:
    claims: list[Claim] = field(default_factory=list)
    summary: str = ""
    refusal_reason: Optional[str] = None


@dataclass
class Decision:
    status: str
    text: str
    missing_fact: Optional[str] = None
    refusal_reason: Optional[str] = None


@dataclass
class SourceCard:
    fragment_id: str
    document_id: str
    document_title: str
    revision_label: str
    clause: str
    quote: str
    line_start: int
    line_end: int


@dataclass
class Answer:
    status: str
    text: str
    claims: list[Claim] = field(default_factory=list)
    sources: list[SourceCard] = field(default_factory=list)
    missing_fact: Optional[str] = None
    refusal_reason: Optional[str] = None


class KnowledgeStore(Protocol):
    def get_fragment(self, fragment_id: str) -> Optional[Fragment]: ...

    def get_document_text(self, document_id: str) -> Optional[str]: ...


@dataclass
class Deps:
    store: KnowledgeStore
    search: Callable[[QueryFacts, int], list[Fragment]]
    select: Callable[[list[Fragment], QueryFacts], Selection]
    extract_facts: Callable[[list[Message]], QueryFacts]
    draft: Callable[[list[Message], QueryFacts, list[Fragment]], Draft]
    verify: Callable[[list[Claim], dict[str, Fragment]], list[Claim]]
    decide: Callable[[QueryFacts, Selection, Draft, list[Claim]], Decision]
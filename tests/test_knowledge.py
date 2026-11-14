from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import date
from hashlib import sha256
from pathlib import Path

import pytest

from assistant_core.contracts import EventDate, QueryFacts, TechnicalError
from assistant_core.knowledge_store import SQLiteKnowledgeStore, open_store
from assistant_core.retrieval import search
from assistant_core.versions import select
from scripts.build_knowledge import build

RAW = Path(__file__).resolve().parents[1] / "data/raw"


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> tuple[SQLiteKnowledgeStore, Path]:
    if not RAW.is_dir():
        pytest.skip("Для проверки корпуса распакуйте выданные TXT в data/raw")
    output = tmp_path_factory.mktemp("knowledge")
    report = build(RAW, output)
    assert report["published"], report["errors"]
    return open_store(output / "knowledge.sqlite"), output


def test_corpus_build_and_citations(corpus: tuple[SQLiteKnowledgeStore, Path], tmp_path: Path) -> None:
    store, output = corpus
    report = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert len(report["sources"]) == 21
    assert report["database_sha256"] == sha256((output / "knowledge.sqlite").read_bytes()).hexdigest()
    rebuilt = build(RAW, tmp_path)
    assert rebuilt == report
    ids = store.read("SELECT fragment_id FROM fragments ORDER BY fragment_id")
    other = open_store(tmp_path / "knowledge.sqlite")
    assert ids == other.read("SELECT fragment_id FROM fragments ORDER BY fragment_id")
    for row in store.read("SELECT document_id, source_path FROM documents"):
        expected = (RAW / row["source_path"]).read_bytes().decode("utf-8-sig")
        assert store.get_document_text(row["document_id"]) == expected
    for row in ids:
        fragment = store.get_fragment(row["fragment_id"])
        assert fragment is not None
        lines = (RAW / fragment.source_path).read_bytes().decode("utf-8-sig").splitlines(keepends=True)
        assert fragment.text == "".join(lines[fragment.line_start - 1:fragment.line_end]).rstrip("\r\n")
    assert store.get_document_text("missing") is None
    assert store.get_fragment("missing") is None


@pytest.mark.parametrize(("counterparty", "number", "clause"), [
    ("Альфа-Трейд", "П-17/2024", "6.2"),
    ("Альфа-Трейдинг", "ПТ-05/2023", "6.1"),
])
def test_structured_party_and_clause(
    corpus: tuple[SQLiteKnowledgeStore, Path], counterparty: str, number: str, clause: str,
) -> None:
    store, _ = corpus
    query = QueryFacts(f"Что указано для ООО «{counterparty}» в п. {clause}?")
    assert [(f.document_title.split("№ ")[-1], f.clause) for f in search(store, query)] == [(number, clause)]


def test_structured_contract_and_statute(corpus: tuple[SQLiteKnowledgeStore, Path]) -> None:
    store, _ = corpus
    candidates = search(store, QueryFacts("№ П‑17/2024, пункт 2.1"))
    assert [(f.clause, f.line_start) for f in candidates] == [("2.1", 10)]
    assert search(store, QueryFacts("П-99/2024, п. 2.1")) == []
    candidates = search(store, QueryFacts("Статья 13.11 КоАП РФ", document_refs=["13.11"]))
    assert candidates and all(f.clause.startswith("ст. 13.11") for f in candidates)
    candidates = search(store, QueryFacts("п. 2 ст. 9 152-ФЗ"))
    assert len(candidates) == 1 and candidates[0].line_start == 13


@pytest.mark.parametrize(("when", "revision", "threshold"), [
    (date(2025, 2, 28), "Редакция от 10.01.2023", "10 000 000"),
    (date(2025, 3, 1), "Редакция от 01.03.2025", "15 000 000"),
])
def test_regulation_boundary(
    corpus: tuple[SQLiteKnowledgeStore, Path], when: date, revision: str, threshold: str,
) -> None:
    store, _ = corpus
    query = QueryFacts("Регламент согласования договоров, п. 2.4", dates=[EventDate("deal", when)])
    selected = select(store, search(store, query, 1), query)
    assert not selected.gaps
    assert [f.revision_label for f in selected.applicable] == [revision]
    assert threshold in selected.applicable[0].text


@pytest.mark.parametrize(("when", "revision", "start"), [
    (date(2024, 9, 30), "Исходная редакция", 14),
    (date(2024, 10, 1), "Исходная редакция", 14),
    (date(2024, 10, 2), "Допсоглашение № 1", 6),
])
def test_specification_boundary(
    corpus: tuple[SQLiteKnowledgeStore, Path], when: date, revision: str, start: int,
) -> None:
    store, _ = corpus
    query = QueryFacts("П-17/2024, п. 3.2", dates=[EventDate("spec_signed", when)])
    selected = select(store, search(store, query, 1), query)
    assert not selected.gaps
    assert [(f.revision_label, f.line_start) for f in selected.applicable] == [(revision, start)]


def test_delivery_requires_specification_date(corpus: tuple[SQLiteKnowledgeStore, Path]) -> None:
    store, _ = corpus
    query = QueryFacts("П-17/2024. Доставка 02.10.2024. Какой адрес?", document_refs=["П-17/2024"],
                       dates=[EventDate("delivery", date(2024, 10, 2))])
    candidates = search(store, query)
    assert any(f.clause == "3.2" for f in candidates)
    selected = select(store, candidates, query)
    assert selected.missing_fact == "дата подписания спецификации"
    assert selected.gaps and all(f.clause != "3.2" for f in selected.applicable)
    query.document_refs.append("п. 2.1")
    selected = select(store, search(store, query), query)
    assert selected.missing_fact is None and [f.clause for f in selected.applicable] == ["2.1"]


@pytest.mark.parametrize(("question", "expected"), [
    ("За сколько дней после поставки нужно проверить качество?", {"3.3", "6.1"}),
    ("Доставка и упаковка уже включены в цену или их оплачиваем отдельно?", {"2.1"}),
])
def test_delivery_topics_without_address(
    corpus: tuple[SQLiteKnowledgeStore, Path], question: str, expected: set[str],
) -> None:
    store, _ = corpus
    facts = QueryFacts(question, document_refs=["П-17/2024"])
    selected = select(store, search(store, facts), facts)
    assert selected.missing_fact is None
    assert expected <= {fragment.clause for fragment in selected.applicable}


@pytest.mark.parametrize(("question", "reference", "search_query", "expected"), [
    ("Грузовик приезжает в два часа ночи. Нас должны пропускать?", "А-08/2024",
     "круглосуточный доступ транспорта", "4.3"),
    ("Можно просто сдать ключи или обязательно оформлять акт?", "А-08/2024",
     "возврат помещения по акту", "6.3"),
    ("В юротдел нужно отнести оригинал сразу или есть какой-то срок?", "регламент согласования договоров",
     "передача подписанного оригинала договора в юридический отдел", "3.2"),
])
def test_organizer_questions_with_search_queries(
    corpus: tuple[SQLiteKnowledgeStore, Path], question: str, reference: str, search_query: str, expected: str,
) -> None:
    store, _ = corpus
    facts = QueryFacts(question, document_refs=[reference], search_queries=[search_query])
    result = select(store, search(store, facts), facts)
    assert result.missing_fact is None
    assert expected in {fragment.clause for fragment in result.applicable}


@pytest.mark.parametrize(("question", "when"), [
    ("Регламент согласования договоров, п. 2.4", date(2023, 1, 9)),
    ("ТК РФ, ст. 70, испытательный срок главного бухгалтера", date(2025, 8, 31)),
])
def test_unavailable_history(corpus: tuple[SQLiteKnowledgeStore, Path], question: str, when: date) -> None:
    store, _ = corpus
    query = QueryFacts(question, dates=[EventDate("deal", when)])
    selected = select(store, search(store, query), query)
    assert not selected.applicable and selected.missing_fact is None
    assert selected.gaps and "В базе нет применимой редакции" in selected.gaps[0]


def test_incomplete_build_keeps_previous_catalogue(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    source = raw / "sample.txt"
    source.write_bytes("ДОГОВОР\r\n1. Условия\r\n1.1. Условие.\r\nПродолжение  \r\n".encode())
    output = tmp_path / "output"
    assert build(raw, output, expected_count=1)["published"]
    database = output / "knowledge.sqlite"
    previous = database.read_bytes()
    clause = search(open_store(database), QueryFacts("п. 1.1"))[0]
    assert clause.text == "1.1. Условие.\r\nПродолжение  "
    source.write_bytes(b"\xff")
    report = build(raw, output, expected_count=1)
    assert not report["published"] and report["sources"][0]["error"]
    assert database.read_bytes() == previous


def test_database_and_index_failures(tmp_path: Path) -> None:
    database = tmp_path / "missing.sqlite"
    with pytest.raises(TechnicalError):
        open_store(database)
    assert not database.exists()
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "sample.txt").write_text("ДОГОВОР\n1.1. Условие.", encoding="utf-8")
    assert build(raw, tmp_path, expected_count=1)["published"]
    database = tmp_path / "knowledge.sqlite"
    store = open_store(database)
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("DROP TABLE fragment_fts")
    with pytest.raises(TechnicalError):
        open_store(database)
    with pytest.raises(TechnicalError):
        search(store, QueryFacts("Поисковой запрос"))

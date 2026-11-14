"""Parse the organizer TXT export while preserving source line coordinates."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from assistant_core.knowledge_store import contract_numbers, normalize, search_text

CLAUSE = re.compile(r"^«?(\d+(?:\.\d+)+)\.\s+")
ARTICLE = re.compile(r"^Статья (\d+(?:[.\-]\d+)*)\.\s+")
PROVISION = re.compile(r"^(\d+(?:[.\-]\d+)*)[.)]\s+")
MONTHS = "января февраля марта апреля мая июня июля августа сентября октября ноября декабря".split()
DATE = r"\d{1,2}\.\d{2}\.\d{4}|\d{1,2}\s+[а-я]+\s+\d{4}"
EDITORIAL = re.compile(
    r"^(?:(?:Статья|Пункт|Часть|Подпункт|Абзац) .+? (?:измен[её]н[аы]?|дополнен[аы]?|"
    r"изложен[аы]? в новой редакции)|Федеральным законом|О проверке конституционности|"
    r"См\.|Примечания изменены|Положения .+ \(в редакции Федерального закона)", re.IGNORECASE
)


@dataclass(frozen=True)
class Passage:
    clause: str
    line_start: int
    line_end: int
    context: str = ""
    effective_from: date | None = None
    application_from: date | None = None


@dataclass(frozen=True)
class Document:
    family: str
    title: str
    revision: str
    text: str
    passages: list[Passage]
    refs: set[tuple[str, str]]
    kind: str = "standalone"
    valid_from: date | None = None
    valid_to: date | None = None
    event_type: str = ""


def stable_id(prefix: str, *parts: str) -> str:
    key = "\0".join(parts).encode()
    return f"{prefix}-{hashlib.sha256(key).hexdigest()[:24]}"


def parse_date(value: str) -> date:
    if "." in value:
        day, month, year = map(int, value.split("."))
    else:
        day_text, month_text, year_text = value.casefold().split()
        day, month, year = int(day_text), MONTHS.index(month_text) + 1, int(year_text)
    return date(year, month, day)


def find_date(pattern: str, text: str) -> date | None:
    match = re.search(pattern + f"({DATE})", text, re.IGNORECASE)
    return parse_date(match[1]) if match else None


def numbered_passages(lines: list[str]) -> list[Passage]:
    passages = []
    section = ""
    for index, line in enumerate(lines):
        heading = re.match(r"^\d+\.\s+(.+)", line)
        match = CLAUSE.match(line)
        if heading:
            section = heading[1]
        if match:
            end = index + 1
            while end < len(lines) and lines[end].strip() and not re.match(r"^«?\d+\.", lines[end]):
                end += 1
            passages.append(Passage(match[1], index + 1, end, section))
    return passages


def parse_law(path: Path, text: str, lines: list[str], export: re.Match[str]) -> Document:
    code, article = export.groups()
    start = next((i for i, line in enumerate(lines) if (match := ARTICLE.match(line)) and match[1] == article), None)
    if start is None:
        raise ValueError(f"Не найден заголовок статьи {article}")
    snapshot = find_date(r"Дата выгрузки:\s*", "\n".join(lines[:start]))
    if snapshot is None:
        raise ValueError("Нет даты выгрузки закона")
    # A notice of a future amendment does not date the supplied wording.
    article_dates = [
        change for line in lines[:start] if EDITORIAL.match(line)
        if (change := find_date(r"\bс\s+", line)) and change <= snapshot
    ]
    article_from = max(article_dates, default=None)
    passages = []
    provision = ""
    provision_from = None
    pending_from = None
    notes = False
    single_line = not any(line.strip() for line in lines[start + 1:])
    for i, line in enumerate(lines[start:], start + 1):
        if not line.strip():
            continue
        if EDITORIAL.match(line) and i > start + 1:
            change = find_date(r"\bс\s+", line)
            if change and change <= snapshot:
                pending_from = change
            continue
        if line == "Примечания:":
            notes = True
            provision = ""
            provision_from = pending_from
            pending_from = None
            continue
        match = PROVISION.match(line)
        if match and line[match.end() - 2] == ".":
            provision = match[1]
            if not notes:
                provision_from = pending_from
        effective = max(filter(None, (article_from, provision_from, pending_from)), default=None)
        clause = f"ст. {article}"
        if provision:
            clause += f", примечание {provision}" if notes else f", п. {provision}"
        # A heading without article text is context, not a quotable provision.
        if i != start + 1 or single_line or code == "МРОТ":
            passages.append(Passage(clause, i, i, effective_from=effective))
        pending_from = None
    code_key = normalize(code)
    return Document(path.stem, f"{code} — {lines[start]}", f"Выгрузка от {snapshot:%d.%m.%Y}", text,
                    passages, {("article", article), ("code", code_key)}, kind="law",
                    valid_to=snapshot + timedelta(days=1))


def parse_document(path: Path, raw: bytes) -> Document:
    text = raw.decode("utf-8-sig")
    lines = text.splitlines()
    if not lines or not lines[0].strip():
        raise ValueError("Нет заголовка документа")

    export = re.fullmatch(r"законы__(.+)_ст([\d.\-]+)_действующая", path.stem)
    if export:
        return parse_law(path, text, lines, export)

    header = "\n".join(lines[:5])
    parties = {("party", search_text(name)) for name in re.findall(r"ООО\s+«([^»]+)»", header)}
    numbers = contract_numbers("\n".join(lines[:2]))
    if numbers:
        if len(numbers) != 1:
            raise ValueError("Неоднозначный номер договора")
        number = numbers.pop()
        refs = parties | {("contract", number)}
        if lines[0].startswith("ДОПОЛНИТЕЛЬНОЕ СОГЛАШЕНИЕ"):
            signed = find_date(r"—\s*", lines[2])
            passages = []
            for i, line in enumerate(lines):
                clause = CLAUSE.match(line)
                if clause is None:
                    continue
                condition = find_date(r"Спецификации .+ подписаны после\s+", "".join(lines[i + 1:i + 2]))
                if signed is None or condition is None:
                    raise ValueError("Не распознаны даты допсоглашения и условия применения")
                passages.append(Passage(clause[1], i + 1, i + 2, "Порядок поставки", signed,
                                        condition + timedelta(days=1)))
            agreement = lines[0].split("№", 1)[1].strip()
            return Document(number, lines[0], f"Допсоглашение № {agreement}",
                            text, passages, refs, kind="amendment", event_type="spec_signed")
        return Document(number, lines[0], "Исходная редакция", text, numbered_passages(lines), refs,
                        valid_from=find_date(r"—\s*", lines[1]))

    regulation = next((line for line in lines[:3] if line.startswith("РЕГЛАМЕНТ")), None)
    if regulation:
        revision = find_date(r"^Редакция от\s+", " ".join(line for line in lines if line.startswith("Редакция от")))
        if revision is None:
            raise ValueError("Не распознана дата редакции регламента")
        return Document(normalize(regulation), regulation, f"Редакция от {revision:%d.%m.%Y}", text,
                        numbered_passages(lines), {("title", "регламент согласования договоров")},
                        kind="regulation", valid_from=find_date(r"Действует с\s+", header) or revision,
                        valid_to=find_date(r"УТРАТИЛА СИЛУ с\s+", lines[0]), event_type="deal")

    if "Исх. №" in header:
        title = f"Письмо комплаенс-службы {lines[2]}"
        return Document(path.stem, title, "Выданный текст", text,
                        [Passage(f"стр. {i}", i, i, lines[4]) for i in range(7, 13)],
                        {("title", "письмо комплаенс службы"), ("title", "к 12")},
                        valid_from=find_date(r"^С\s+", lines[7]), event_type="deal")
    if "СЛУЖЕБНАЯ ЗАПИСКА" in header:
        return Document(path.stem, lines[2], "Выданный текст", text,
                        [Passage(f"стр. {i}", i, i, lines[3]) for i in range(5, 9)],
                        {("title", "служебная записка"), ("title", "юо 7")},
                        valid_from=find_date(r"\bот\s+", lines[2]), event_type="deal")
    return Document(path.stem, lines[0], "Выданный текст", text, numbered_passages(lines), set())

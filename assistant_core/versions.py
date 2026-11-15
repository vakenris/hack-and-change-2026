"""Select the applicable source wording for each retrieved provision."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from datetime import date
import re
from sqlite3 import Row

from .contracts import Fragment, QueryFacts, Selection
from .knowledge_store import FRAGMENT_COLUMNS, SQLiteKnowledgeStore, normalize
from .retrieval import clause_numbers

FACT_NAMES = {
    "spec_signed": "дата подписания спецификации",
    "deal": "дата заключения договора",
    "as_of": "дата, на которую нужно проверить редакцию",
}
MONTH_ROOTS = "январ феврал март апрел ма июн июл август сентябр октябр ноябр декабр".split()
MONTH_NAME = rf"({'|'.join(MONTH_ROOTS)})[а-я]{{0,3}}\s+(\d{{4}})"
MONTH_COMPARISON = re.compile(rf"\b{MONTH_NAME}(?:\s+года?)?\s+и\s+(?:в\s+)?{MONTH_NAME}\b")


def active(row: Row, when: str) -> bool:
    start = row["valid_from"]
    end = row["valid_to"]
    return (start is None or start <= when) and (end is None or when < end)


def choose_revision(documents: list[Row], when: str) -> Row | None:
    return max((row for row in documents if active(row, when)),
               key=lambda row: row["valid_from"] or "", default=None)


def fragment_at(
    versions: list[Row], rows: list[Row], amendments: list[Row], when: str | None,
) -> tuple[Row | None, Row | None]:
    revision = choose_revision(versions, when) if when else max(
        versions, key=lambda row: row["valid_from"] or "", default=None
    )
    chosen = next((row for row in rows if revision and row["document_id"] == revision["document_id"]), None)
    if when:
        chosen = next((row for row in amendments if row["application_from"] <= when
                       and row["effective_from"] <= when), chosen)
        if chosen is not None and chosen["effective_from"] and chosen["effective_from"] > when:
            chosen = None
    return revision, chosen


def dates_to_check(dates: set[str], versions: list[Row], rows: list[Row]) -> list[str | None]:
    if not dates:
        return [None]
    start, end = min(dates), max(dates)
    # Month bounds and ambiguous dates can enclose a change of wording.
    boundaries = {
        *(row[key] for row in versions for key in ("valid_from", "valid_to")),
        *(row[key] for row in rows for key in ("effective_from", "application_from")),
    }
    return sorted(dates | {day for day in boundaries if day and start < day < end})


def select(store: SQLiteKnowledgeStore, candidates: list[Fragment], facts: QueryFacts) -> Selection:
    groups = defaultdict(list)
    for event in facts.dates:
        groups[event.period].append(event)
    # A coordinated pair of named months is a comparison even when
    # extraction recognises the dates but omits their optional labels.
    if set(groups) == {""}:
        match = MONTH_COMPARISON.search(normalize(facts.question))
        if match:
            first, first_year, second, second_year = match.groups()
            months = {(int(first_year), MONTH_ROOTS.index(first) + 1),
                      (int(second_year), MONTH_ROOTS.index(second) + 1)}
            dates = groups.pop("")
            for event in dates:
                month = (event.value.year, event.value.month)
                period = event.value.strftime("%Y-%m") if month in months and event.event_type in (
                    "as_of", "deal", "unknown"
                ) else ""
                groups[period].append(event)
    if len(groups.keys() - {""}) < 2:
        return select_period(store, candidates, facts)

    result = Selection()
    applicable: dict[str, Fragment] = {}
    missing = set()
    for period, dates in groups.items():
        if not period:
            continue
        start = min(event.value for event in dates)
        end = max(event.value for event in dates)
        label = f"{start:%d.%m.%Y}" if start == end else f"{start:%d.%m.%Y}–{end:%d.%m.%Y}"
        selected = select_period(store, candidates, replace(facts, dates=[*dates, *groups.get("", [])]))
        for fragment in selected.applicable:
            existing = applicable.get(fragment.fragment_id, fragment)
            applicable[fragment.fragment_id] = replace(existing, periods=(*existing.periods, label))
        result.gaps.extend(f"Период {label}. {gap}" for gap in dict.fromkeys(selected.gaps))
        if selected.missing_fact:
            missing.add(selected.missing_fact)
    result.applicable = list(applicable.values())
    result.missing_fact = ", ".join(sorted(missing)) or None
    return result


def select_period(store: SQLiteKnowledgeStore, candidates: list[Fragment], facts: QueryFacts) -> Selection:
    result = Selection()
    if not candidates:
        return result
    dates: dict[str, set[str]] = defaultdict(set)
    for event in facts.dates:
        dates[event.event_type].add(event.value.isoformat())

    ids = [fragment.fragment_id for fragment in candidates]
    placeholders = ",".join("?" for _ in ids)
    seeds = {
        row["fragment_id"]: row for row in store.read(f"""
            SELECT f.fragment_id, f.slot, d.family_id
            FROM fragments f JOIN documents d USING (document_id)
            WHERE f.fragment_id IN ({placeholders})
        """, ids)
    }
    families = sorted({row["family_id"] for row in seeds.values()})
    if not families:
        return result
    explicit_clauses = clause_numbers(facts)
    placeholders = ",".join("?" for _ in families)
    documents: dict[str, list[Row]] = defaultdict(list)
    for row in store.read(f"SELECT * FROM documents WHERE family_id IN ({placeholders})", families):
        documents[row["family_id"]].append(row)

    visited = set()
    required_facts = set()
    for candidate in candidates:
        seed = seeds.get(candidate.fragment_id)
        if seed is None:
            continue
        family, slot = seed["family_id"], seed["slot"]
        if (family, slot) in visited:
            continue
        visited.add((family, slot))

        group = documents[family]
        versions = [row for row in group if row["kind"] != "amendment"]
        versions_event = next((row["event_type"] for row in versions if row["event_type"]), "")
        rows = store.read(f"""
            SELECT {FRAGMENT_COLUMNS}, f.effective_from, f.application_from, d.kind, d.event_type
            FROM fragments f JOIN documents d USING (document_id)
            WHERE d.family_id = ? AND f.slot = ?
            ORDER BY f.application_from DESC, f.effective_from DESC, f.fragment_id
        """, (family, slot))
        amendments = [row for row in rows if row["kind"] == "amendment"]
        event_type = amendments[0]["event_type"] if amendments else versions_event
        if not amendments and dates.get("as_of"):
            event_type = "as_of"
        if event_type:
            relevant_dates = dates.get(event_type, set())
        else:
            # Law exports contain a snapshot. Explicit event dates can reveal a gap in it.
            relevant_dates = set().union(*(values for kind, values in dates.items() if kind != "unknown"))
        label = f"{candidate.document_title}, {candidate.clause}"
        fact = FACT_NAMES.get(event_type, "дата события, к которому относится вопрос")
        is_law = any(row["kind"] == "law" for row in versions)
        if not relevant_dates and (amendments or versions_event == "deal" and dates):
            required_facts.add(fact)
            result.gaps.append(f"Для выбора применимой редакции ({label}) нужна {fact}.")
            continue

        # An unclassified date is enough only if the catalogue confirms the
        # same wording throughout the period it could refer to.
        unknown_law_dates = dates.get("unknown") if is_law and not relevant_dates else None
        checks = dates_to_check(unknown_law_dates or relevant_dates, versions, rows)
        selections = [fragment_at(versions, rows, amendments, when) for when in checks]
        fragment_ids = {row["fragment_id"] if row is not None else None for _, row in selections}
        if unknown_law_dates and None in fragment_ids:
            required_facts.add(fact)
            result.gaps.append(f"Для выбора применимой редакции ({label}) нужна {fact}.")
            continue
        if len(fragment_ids) > 1:
            required_facts.add(fact)
            result.gaps.append(f"В указанные даты применяются разные положения ({label}). Нужна точная {fact}.")
            continue
        revision, chosen = selections[0]
        if chosen is None:
            # A provision from another edition may be only an incidental search match.
            if revision is None or is_law or candidate.clause in explicit_clauses:
                start, end = checks[0], checks[-1]
                at = ""
                if start and end:
                    first, last = date.fromisoformat(start), date.fromisoformat(end)
                    at = f" на {first:%d.%m.%Y}" if first == last else f" за период {first:%d.%m.%Y}–{last:%d.%m.%Y}"
                result.gaps.append(f"В базе нет применимой редакции{at} ({label}).")
            continue
        result.applicable.append(Fragment(**{
            key: chosen[key] for key in Fragment.__dataclass_fields__ if key != "periods"
        }))

    if required_facts:
        result.missing_fact = ", ".join(sorted(required_facts))
    return result

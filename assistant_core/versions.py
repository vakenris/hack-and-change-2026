"""Select the applicable source wording for each retrieved provision."""
from __future__ import annotations

from collections import defaultdict
from datetime import date
from sqlite3 import Row

from .contracts import Fragment, QueryFacts, Selection
from .knowledge_store import FRAGMENT_COLUMNS, SQLiteKnowledgeStore

FACT_NAMES = {"spec_signed": "дата подписания спецификации", "deal": "дата заключения договора"}


def active(row: Row, when: str) -> bool:
    start = row["valid_from"]
    end = row["valid_to"]
    return (start is None or start <= when) and (end is None or when < end)


def choose_revision(documents: list[Row], when: str) -> Row | None:
    return max((row for row in documents if active(row, when)),
               key=lambda row: row["valid_from"] or "", default=None)


def select(store: SQLiteKnowledgeStore, candidates: list[Fragment], facts: QueryFacts) -> Selection:
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
        if event_type:
            relevant_dates = dates.get(event_type, set())
        else:
            # Law exports contain a snapshot. Explicit event dates can reveal a gap in it.
            relevant_dates = set().union(*(values for kind, values in dates.items() if kind != "unknown"))
        label = f"{candidate.document_title}, {candidate.clause}"
        fact = FACT_NAMES.get(event_type, "дата события, к которому относится вопрос")
        if len(relevant_dates) > 1 or (not relevant_dates and (amendments or versions_event == "deal" and dates)):
            required_facts.add(fact)
            result.gaps.append(f"Для выбора применимой редакции ({label}) нужна {fact}.")
            continue

        when = next(iter(relevant_dates), None)
        # For an undated question use the latest supplied wording; the corpus has its own cutoff.
        revision = choose_revision(versions, when) if when else max(
            versions, key=lambda row: row["valid_from"] or "", default=None
        )
        chosen = next((row for row in rows if revision and row["document_id"] == revision["document_id"]), None)
        if when and amendments:
            chosen = next((row for row in amendments if row["application_from"] <= when
                           and row["effective_from"] <= when), chosen)
        if chosen is None or when and chosen["effective_from"] and chosen["effective_from"] > when:
            if revision is None or revision["kind"] == "law" or when and revision["valid_to"] is not None:
                at = f" на {date.fromisoformat(when):%d.%m.%Y}" if when else ""
                result.gaps.append(f"В базе нет применимой редакции{at} ({label}).")
            continue
        result.applicable.append(Fragment(**{key: chosen[key] for key in Fragment.__dataclass_fields__}))

    if required_facts:
        result.missing_fact = ", ".join(sorted(required_facts))
    return result

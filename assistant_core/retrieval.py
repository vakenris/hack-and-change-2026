"""Structured references take precedence over lexical matching."""
from __future__ import annotations

import re
from collections import defaultdict

from .contracts import Fragment, QueryFacts
from .knowledge_store import FRAGMENT_COLUMNS, SQLiteKnowledgeStore, contract_numbers, normalize, search_text

CLAUSE_REF = re.compile(r"(?<!\w)(?:пп?\.|пункт(?:а|у|ом|е|ы|ов)?)\s*(\d+(?:\.\d+)*(?:-\d+)?)\b")
ARTICLE_REF = re.compile(r"(?<!\w)(?:ст\.|стать(?:я|и|ю|е))\s*(\d+(?:[.\-]\d+)*)\b")
BARE_CLAUSE = re.compile(r"\d+(?:\.\d+)+")
STOP_WORDS = frozenset(
    "а без бы в во для до за и из или к ко ли на не но о об от по под при с со у что это я мы как "
    "какой какая какое можно нужно должны можем договор договора договору договором руб рублей п ст рф ооо".split()
)


def clause_numbers(facts: QueryFacts) -> set[str]:
    references = "\n".join(facts.document_refs)
    explicit = normalize("\n".join((facts.question, references)))
    fallback = normalize("\n".join(facts.search_queries))
    articles = set(ARTICLE_REF.findall(explicit) or ARTICLE_REF.findall(fallback))
    bare = {ref.strip() for ref in facts.document_refs if BARE_CLAUSE.fullmatch(ref.strip())}
    return set(CLAUSE_REF.findall(explicit) or CLAUSE_REF.findall(fallback)) | (bare - articles)


def match_expression(text: str) -> str:
    terms = dict.fromkeys(
        term for term in search_text(text).split() if term not in STOP_WORDS and len(term) > 1 and not term.isdecimal()
    )
    expressions = []
    for term in list(terms)[:48]:
        expressions.append(f'"{term}"')
        if len(term) >= 6 and term.isalpha():
            expressions.append(f'"{term[:max(5, len(term) - 3)]}"*')
        elif term in {"акт", "срок"}:
            expressions.append(f'"{term}"*')
    return " OR ".join(dict.fromkeys(expressions))


def search(store: SQLiteKnowledgeStore, facts: QueryFacts, limit: int = 12) -> list[Fragment]:
    if limit <= 0:
        return []
    references = "\n".join(facts.document_refs)
    text = "\n".join([facts.question, references, *facts.search_queries])
    explicit_text = normalize("\n".join((facts.question, references)))
    fallback_text = normalize(text)
    numbers = contract_numbers(references) or contract_numbers(facts.question) or contract_numbers(text)
    articles = set(ARTICLE_REF.findall(explicit_text) or ARTICLE_REF.findall(fallback_text))
    clauses = clause_numbers(facts)

    refs: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in store.read("SELECT family_id, kind, value FROM refs"):
        refs[row["kind"]][row["value"]].add(row["family_id"])

    families: set[str] | None = None
    if numbers or articles:
        families = set().union(*(refs["contract"].get(number, set()) for number in numbers))
        law_families = set().union(*(refs["article"].get(article, set()) for article in articles))
        padded = f" {search_text(text)} "
        codes = [code for code in refs["code"] if f" {search_text(code)} " in padded]
        if codes:
            law_families.intersection_update(set().union(*(refs["code"][code] for code in codes)))
        families.update(law_families)
    else:
        for source in ("\n".join((facts.question, references)), text):
            padded = f" {search_text(source)} "
            matches = [value for value in refs["party"] if f" {value} " in padded and value != "меридиан"]
            matches += [value for value in refs["title"] if f" {value} " in padded]
            if matches:
                families = set().union(*(refs[kind].get(value, set()) for kind in ("party", "title") for value in matches))
                break

    if families is not None and not families:
        return []
    conditions = []
    parameters: list[str | int] = []
    if families is not None:
        conditions.append("d.family_id IN (" + ",".join("?" for _ in families) + ")")
        parameters.extend(sorted(families))
    if clauses:
        placeholders = ",".join("?" for _ in clauses)
        conditions.append(
            f"(f.clause IN ({placeholders}) OR (d.kind = 'law' AND instr(f.clause, ', п. ') > 0 "
            f"AND substr(f.clause, instr(f.clause, ', п. ') + 5) IN ({placeholders})))"
        )
        parameters.extend(sorted(clauses) * 2)
    else:
        query_text = search_text(" ".join((facts.question, *facts.search_queries)))
        asks_address = any(phrase in query_text for phrase in ("адрес", "куда", "место поставки",
                                                              "место доставки", "порядок поставки"))
        if not asks_address and not any(event.event_type == "spec_signed" for event in facts.dates):
            conditions.append("NOT EXISTS (SELECT 1 FROM documents a JOIN fragments af USING (document_id) "
                              "WHERE a.kind = 'amendment' AND a.event_type = 'spec_signed' "
                              "AND a.family_id = d.family_id AND af.slot = f.slot)")
    where = " WHERE " + " AND ".join(conditions) if conditions else ""
    select = f"SELECT {FRAGMENT_COLUMNS} FROM fragments f JOIN documents d USING (document_id)"
    query = match_expression(text)
    if not query and families is None and not clauses:
        return []
    if clauses or not query:
        rows = store.read(select + where + " ORDER BY f.source_path, f.line_start LIMIT ?", (*parameters, limit))
    else:
        operator = " AND " if where else " WHERE "
        rows = store.read(
            select + " JOIN fragment_fts ON fragment_fts.rowid = f.rowid" + where + operator
            + "fragment_fts MATCH ? ORDER BY bm25(fragment_fts, 1.0, 0.3), f.source_path, f.line_start LIMIT ?",
            (*parameters, query, limit),
        )
        if not rows and families is not None:
            rows = store.read(select + where + " ORDER BY f.source_path, f.line_start LIMIT ?", (*parameters, limit))
    return [Fragment(**dict(row)) for row in rows]

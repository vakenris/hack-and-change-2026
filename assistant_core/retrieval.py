"""Structured references take precedence over lexical matching."""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import replace
from itertools import zip_longest

from .contracts import Fragment, QueryFacts
from .knowledge_store import FRAGMENT_COLUMNS, SQLiteKnowledgeStore, contract_numbers, normalize, search_text

PROVISION_NUMBER = r"\s*(\d+(?:\.\d+)*(?:-\d+)?)\b"
CLAUSE_PATTERN = r"(?<!\w)(?:пп?\.|пункт(?:а|у|ом|е|ы|ов)?)" + PROVISION_NUMBER
ARTICLE_PATTERN = r"(?<!\w)(?:ст\.|стать(?:я|и|ю|е))\s*(\d+(?:[.\-]\d+)*)\b"
CLAUSE_REF = re.compile(CLAUSE_PATTERN)
ARTICLE_REF = re.compile(ARTICLE_PATTERN)
LAW_CLAUSE_REF = re.compile(
    rf"(?:{CLAUSE_PATTERN}|(?<!\w)(?:ч\.|част[ьи]){PROVISION_NUMBER})[^\S\r\n]+{ARTICLE_PATTERN}"
)
BARE_CLAUSE = re.compile(r"\d+(?:\.\d+)+")
STOP_WORDS = frozenset(
    "а без бы в во для до за и из или к ко ли на не но о об от по под при с со у что это я мы как "
    "какой какая какое можно нужно должны можем договор договора договору договором руб рублей п ст рф ооо".split()
)
QUERY_EXPANSIONS = (
    (re.compile(r"\bаванс\w*\b"), "предоплата"),
    (re.compile(r"\bюротдел\w*\b"), "юридический отдел"),
    (re.compile(r"\bпропуск\w*\b"), "доступ"),
)
DEFAULT_RATE = re.compile(r"\b(?:основн\w*|общ\w*|стандартн\w*)\s+(?:налогов\w*\s+)?ставк\w*")
QUESTION_BOUNDARY = re.compile(
    r"(?<=[?;])\s+|,?\s+(?:и|а также)\s+(?=(?:как\w*|сколько|когда|нужн\w*|требу\w*|"
    r"вход\w*|можно|допускается)\b)", re.IGNORECASE,
)


def clause_numbers(facts: QueryFacts) -> set[str]:
    references = "\n".join(facts.document_refs)
    explicit = normalize("\n".join((facts.question, references)))
    fallback = normalize("\n".join(facts.search_queries))
    articles = set(ARTICLE_REF.findall(explicit) or ARTICLE_REF.findall(fallback))
    bare = {ref.strip() for ref in facts.document_refs if BARE_CLAUSE.fullmatch(ref.strip())}
    return set(CLAUSE_REF.findall(explicit) or CLAUSE_REF.findall(fallback)) | (bare - articles)


def match_expression(text: str) -> str:
    text = search_text(text)
    additions = " ".join(term for pattern, term in QUERY_EXPANSIONS if pattern.search(text))
    terms = dict.fromkeys(
        term for term in f"{additions} {text}".split()
        if term not in STOP_WORDS and len(term) > 1 and not term.isdecimal()
    )
    expressions = []
    if DEFAULT_RATE.search(text):
        expressions.append('"основная ставка"')
    for term in list(terms)[:48]:
        expressions.append(f'"{term}"')
        if len(term) >= 6 and term.isalpha():
            expressions.append(f'"{term[:max(5, len(term) - 3)]}"*')
        elif term in {"акт", "срок"}:
            expressions.append(f'"{term}"*')
    return " OR ".join(dict.fromkeys(expressions))


def ranked_fragments(
    store: SQLiteKnowledgeStore, facts: QueryFacts, text: str,
    families: set[str] | None, clauses: set[str], limit: int,
) -> list[Fragment]:
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
            + "fragment_fts MATCH ? ORDER BY "
            + ("(instr(fragment_fts.context, 'основная ставка') > 0) DESC, "
               if DEFAULT_RATE.search(search_text(text)) else "")
            + "bm25(fragment_fts, 1.0, 0.3), f.source_path, f.line_start LIMIT ?",
            (*parameters, query, limit),
        )
        if not rows and families is not None:
            rows = store.read(select + where + " ORDER BY f.source_path, f.line_start LIMIT ?", (*parameters, limit))
    return [Fragment(**dict(row)) for row in rows]


def catalogue_refs(store: SQLiteKnowledgeStore) -> dict[str, dict[str, set[str]]]:
    refs: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in store.read("SELECT family_id, kind, value FROM refs"):
        refs[row["kind"]][row["value"]].add(row["family_id"])
    return refs


def named_party(name: str, padded: str) -> bool:
    return bool(re.search(rf" {re.escape(name)}(?:ом|ем|а|у|е|ы)? ", padded))


def counterparty_clarification(store: SQLiteKnowledgeStore, facts: QueryFacts) -> tuple[str, str] | None:
    if contract_numbers(facts.question):
        return None
    padded = f" {search_text(facts.question)} "
    parties = catalogue_refs(store)["party"]
    aliases: dict[str, set[str]] = defaultdict(set)
    for name, families in parties.items():
        if " " in name:
            aliases[name.split()[0]].update(families)
    words = set(padded.split())
    for alias, families in aliases.items():
        forms = {alias}
        if alias.endswith("а"):
            forms.update(alias[:-1] + ending for ending in ("ы", "е", "у", "ой", "ою"))
        if len(families) > 1 and forms & words:
            if any(name.startswith(alias + " ") and named_party(name, padded) for name in parties):
                continue
            placeholders = ",".join("?" for _ in families)
            rows = store.read(
                f"SELECT family_id, title, text FROM documents WHERE family_id IN ({placeholders}) "
                "AND kind = 'standalone' ORDER BY title", sorted(families),
            )
            if len(rows) > 1:
                titles = []
                for row in rows:
                    names = re.findall(r"ООО\s+«([^»]+)»", "\n".join(row["text"].splitlines()[:8]))
                    party = next((name for name in names if search_text(name).startswith(alias + " ")), "")
                    titles.append(f'{row["title"]} (ООО «{party}»)' if party else row["title"])
                return "контрагент или номер договора", "; ".join(titles)
    return None


def merge_ranked(groups: list[list[Fragment]], limit: int) -> list[Fragment]:
    fragments: dict[str, Fragment] = {}
    for row in zip_longest(*groups):
        for fragment in row:
            if fragment is not None:
                fragments.setdefault(fragment.fragment_id, fragment)
                if len(fragments) == limit:
                    return list(fragments.values())
    return list(fragments.values())


def search(store: SQLiteKnowledgeStore, facts: QueryFacts, limit: int = 12) -> list[Fragment]:
    if limit <= 0:
        return []
    refs = catalogue_refs(store)
    literal_parts = [part.strip() for part in QUESTION_BOUNDARY.split(facts.question) if part.strip()]
    rewrites = list(dict.fromkeys(part.strip() for part in facts.search_queries if part.strip()))
    parts = (rewrites if len(rewrites) > 1 else literal_parts)[:8]
    if len(parts) < 2:
        return search_part(store, facts, refs, limit)

    # Each query carries only its own scope. Recover exact references dropped
    # during rewriting with one additional scoped lookup.
    part_text = "\n".join(parts)
    exact_text = "\n".join((facts.question, *facts.document_refs))
    exact_numbers = contract_numbers(exact_text)
    exact_articles = set(ARTICLE_REF.findall(normalize(exact_text)))
    missing_exact = (exact_numbers - contract_numbers(part_text)
                     or exact_articles - set(ARTICLE_REF.findall(normalize(part_text)))
                     or clause_numbers(replace(facts, search_queries=[])) - set(CLAUSE_REF.findall(normalize(part_text))))
    scoped = [replace(facts, question=part, document_refs=[], search_queries=[]) for part in parts]
    if missing_exact:
        scoped.append(replace(facts, search_queries=[]))
    return merge_ranked([search_part(store, part, refs, limit) for part in scoped], limit)


def search_part(
    store: SQLiteKnowledgeStore, facts: QueryFacts, refs: dict[str, dict[str, set[str]]], limit: int,
) -> list[Fragment]:
    references = "\n".join(facts.document_refs)
    explicit = "\n".join((facts.question, references))
    text = "\n".join((explicit, *facts.search_queries))
    numbers = contract_numbers(references) or contract_numbers(facts.question) or contract_numbers(text)
    articles = set(ARTICLE_REF.findall(normalize(explicit)) or ARTICLE_REF.findall(normalize(text)))
    clauses = clause_numbers(facts)

    named: set[str] = set()
    for source in (explicit, text):
        padded = f" {search_text(source)} "
        padded = re.sub(r"\bсеверн(?:ым|ого|ому|ом) склад(?:ом|а|у|е)\b", "северный склад", padded)
        for kind in ("party", "title"):
            for name, families in refs[kind].items():
                matched = named_party(name, padded) if kind == "party" else f" {name} " in padded
                if name != "меридиан" and matched:
                    named.update(families)
        if named:
            break

    if not numbers and not articles:
        return ranked_fragments(store, facts, text, named or None, clauses, limit)

    # Search exact references independently so a clause of one document
    # cannot filter out the article or clauses of a different document.
    law_clauses: dict[str, set[str]] = defaultdict(set)
    for point, part, article in LAW_CLAUSE_REF.findall(normalize(text)):
        law_clauses[article].add(point or part)
    non_law_clauses = clauses - set().union(*law_clauses.values())
    groups: dict[str, set[str]] = {}
    for number in sorted(numbers):
        for family in sorted(refs["contract"].get(number, set())):
            groups[family] = non_law_clauses
    if articles:
        for family in sorted(named):
            groups.setdefault(family, non_law_clauses)
    padded = f" {search_text(text)} "
    codes = [code for code in refs["code"] if f" {search_text(code)} " in padded]
    code_families = set().union(*(refs["code"][code] for code in codes))
    for article in sorted(articles):
        families = refs["article"].get(article, set())
        if codes:
            families = families & code_families
        for family in sorted(families):
            groups[family] = law_clauses[article] or (clauses if len(articles) == 1 and not (numbers or named) else set())

    ranked = [ranked_fragments(store, facts, text, {family}, scope, limit) for family, scope in groups.items()]
    return merge_ranked(ranked, limit)

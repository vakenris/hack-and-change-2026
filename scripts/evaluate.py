"""Measure evidence coverage on the public questions and local regressions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from collections import Counter
from datetime import date
from pathlib import Path
from time import monotonic

from assistant_core.assistant_service import build_assistant
from assistant_core.contracts import EventDate, QueryFacts, TechnicalError
from assistant_core.knowledge_store import open_store
from assistant_core.retrieval import counterparty_clarification, search
from assistant_core.versions import select
from solution import format_answer, parse_dialogue

ROOT = Path(__file__).resolve().parents[1]


def matches(source, expected: list[str]) -> bool:
    document, clause, *details = expected
    return document.casefold() in source.document_title.casefold() and (
        not clause or source.clause == clause
    ) and all(part.casefold() in (source.text + " " + source.revision_label).casefold()
              for part in details)


def measure(store, row: dict, *, limit: int, assistant=None) -> dict:
    messages = parse_dialogue(row["question"])
    facts = QueryFacts(
        messages[-1].content,
        search_queries=row.get("queries", []),
        dates=[EventDate(kind, date.fromisoformat(value), *period) for kind, value, *period in row.get("dates", [])],
    )
    start = monotonic()
    options = counterparty_clarification(store, facts)
    candidates = [] if options else search(store, facts, limit)
    selection = select(store, candidates, facts)
    expected = row.get("sources", [])
    expected_ranks = [
        next((i for i, source in enumerate(selection.applicable, 1) if matches(source, item)), None)
        for item in expected
    ]
    result = {
        "id": row["id"],
        "evidence_ranks": expected_ranks,
        "missing_fact": options[0] if options else selection.missing_fact,
        "candidate_count": len(candidates),
        "evidence_ms": round((monotonic() - start) * 1000, 2),
    }
    if "expected_status" in row:
        result["expected_status"] = row["expected_status"]
    if assistant:
        start = monotonic()
        try:
            answer = assistant.answer(messages)
            result["status"] = answer.status
            result["answer"] = format_answer(answer)
            cited = [store.get_fragment(source.fragment_id) for source in answer.sources]
            result["cited_expected"] = [
                any(source is not None and matches(source, item) for source in cited) for item in expected
            ]
        except TechnicalError as error:
            result.update(status="error", failed_stage=error.stage, code=error.code)
        result["answer_ms"] = round((monotonic() - start) * 1000, 2)
    return result


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True, help="Public demo_questions.csv")
    parser.add_argument("--database", default="data/processed/knowledge.sqlite")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--attempt-id")
    parser.add_argument("--score", type=float)
    parser.add_argument("--report", type=Path, default=ROOT / "dist/evaluation.json")
    parser.add_argument("--live", action="store_true", help="Also call the configured model to answer the questions")
    args = parser.parse_args()

    fixture = json.loads((ROOT / "tests/fixtures/evaluation.json").read_text(encoding="utf-8"))
    with args.questions.open(encoding="utf-8-sig", newline="") as handle:
        public = list(csv.DictReader(handle))
    if {row["id"] for row in public} != fixture["public"].keys():
        parser.error("questions.csv должен содержать ID из открытого набора организаторов")
    rows = [{**row, **fixture["public"][row["id"]]} for row in public] + fixture["regressions"]
    store = open_store(args.database)
    if args.live and os.environ.get("ASSISTANT_STUB") == "1":
        parser.error("--live требует ASSISTANT_STUB=0")
    assistant = build_assistant(args.database) if args.live else None
    measured = [measure(store, row, limit=12, assistant=assistant) for row in rows]
    summary = {}
    for name, sample in (("public", measured[:len(public)]), ("regressions", measured[len(public):])):
        with_sources = [row for row in sample if row["evidence_ranks"]]
        summary[name] = {
            "questions": len(sample),
            "with_expected_evidence": len(with_sources),
            "all_expected_evidence_found": sum(None not in row["evidence_ranks"] for row in with_sources),
            "all_expected_evidence_top_3": sum(
                all(rank is not None and rank <= 3 for rank in row["evidence_ranks"]) for row in with_sources
            ),
            "statuses": dict(Counter(row["status"] for row in sample if "status" in row)),
            "status_matches": sum(row.get("status") == row["expected_status"]
                                  for row in sample if "expected_status" in row) if args.live else None,
        }
    report = {
        "git_head": git_value("rev-parse", "HEAD"),
        "working_tree_dirty": bool(git_value("status", "--porcelain")),
        "zip_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest() if args.archive else None,
        "attempt_id": args.attempt_id,
        "score": args.score,
        "live": args.live,
        "rerank": args.live and os.environ.get("RERANKER_ENABLED") == "1",
        "summary": summary,
        "cases": measured,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

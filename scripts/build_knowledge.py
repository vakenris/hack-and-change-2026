"""Build with `python -m scripts.build_knowledge` from the repository root."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from assistant_core.knowledge_store import SCHEMA_VERSION, search_text
from scripts.ingestion import Document, parse_document, stable_id

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = """
CREATE TABLE documents (
    document_id TEXT PRIMARY KEY, family_id TEXT NOT NULL, title TEXT NOT NULL,
    revision_id TEXT NOT NULL UNIQUE, revision_label TEXT NOT NULL, text TEXT NOT NULL,
    source_path TEXT NOT NULL UNIQUE, kind TEXT NOT NULL, valid_from TEXT, valid_to TEXT,
    event_type TEXT NOT NULL
);
CREATE INDEX documents_family ON documents(family_id);
CREATE TABLE refs (
    family_id TEXT NOT NULL, kind TEXT NOT NULL, value TEXT NOT NULL,
    PRIMARY KEY(kind, value, family_id)
) WITHOUT ROWID;
CREATE TABLE fragments (
    fragment_id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents,
    document_title TEXT NOT NULL, revision_id TEXT NOT NULL, revision_label TEXT NOT NULL,
    clause TEXT NOT NULL, text TEXT NOT NULL, source_path TEXT NOT NULL,
    line_start INTEGER NOT NULL, line_end INTEGER NOT NULL, slot TEXT NOT NULL,
    effective_from TEXT, application_from TEXT
);
CREATE INDEX fragments_slot ON fragments(document_id, slot);
CREATE VIRTUAL TABLE fragment_fts USING fts5(body, context);
"""


def iso(value: date | None) -> str | None:
    return value.isoformat() if value else None


def insert_document(connection: sqlite3.Connection, path: str, document: Document) -> None:
    document_id = stable_id("doc", path)
    revision_id = stable_id("rev", document.family, path, document.revision)
    connection.execute(
        "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (document_id, document.family, document.title, revision_id, document.revision,
         document.text, path, document.kind, iso(document.valid_from), iso(document.valid_to), document.event_type),
    )
    connection.executemany(
        "INSERT OR IGNORE INTO refs VALUES (?, ?, ?)",
        [(document.family, kind, value) for kind, value in sorted(document.refs)],
    )
    lines = document.text.splitlines(keepends=True)
    for passage in document.passages:
        text = "".join(lines[passage.line_start - 1:passage.line_end]).rstrip("\r\n")
        slot = str(passage.line_start) if document.kind == "law" else passage.clause
        fragment_id = stable_id("frag", revision_id, passage.clause, str(passage.line_start))
        cursor = connection.execute(
            "INSERT INTO fragments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (fragment_id, document_id, document.title, revision_id, document.revision, passage.clause,
             text, path, passage.line_start, passage.line_end, slot,
             iso(passage.effective_from), iso(passage.application_from)),
        )
        connection.execute(
            "INSERT INTO fragment_fts(rowid, body, context) VALUES (?, ?, ?)",
            (cursor.lastrowid, search_text(text), search_text(passage.context)),
        )


def build(raw_dir: Path, output_dir: Path, expected_count: int = 21) -> dict[str, Any]:
    files = sorted(path for path in raw_dir.rglob("*") if path.is_file() and path.suffix.casefold() == ".txt")
    entries = []
    errors = []
    if len(files) != expected_count or not files:
        errors.append(f"Ожидалось {expected_count} TXT, найдено {len(files)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir) as staging:
        database = Path(staging) / "knowledge.sqlite"
        with closing(sqlite3.connect(database)) as connection, connection:
            connection.executescript(SCHEMA)
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            for path in files:
                relative = path.relative_to(raw_dir).as_posix()
                entry = {"source_path": relative}
                entries.append(entry)
                try:
                    raw = path.read_bytes()
                    entry["sha256"] = hashlib.sha256(raw).hexdigest()
                    document = parse_document(path, raw)
                    if not document.passages:
                        raise ValueError("Не найдено ни одного фрагмента")
                except (OSError, ValueError, IndexError) as exc:
                    entry["error"] = str(exc)
                    errors.append(f"{relative}: {exc}")
                    continue
                insert_document(connection, relative, document)
                entry.update(document_id=stable_id("doc", relative), fragments=len(document.passages))
            for row in connection.execute("""
                SELECT DISTINCT a.source_path, f.slot FROM documents a JOIN fragments f USING (document_id)
                WHERE a.kind = 'amendment' AND NOT EXISTS (
                    SELECT 1 FROM documents b JOIN fragments base USING (document_id)
                    WHERE b.family_id = a.family_id AND b.kind = 'standalone' AND base.slot = f.slot
                )
            """):
                errors.append(f"{row[0]}: не найден исходный п. {row[1]}")
            connection.execute("INSERT INTO fragment_fts(fragment_fts) VALUES ('integrity-check')")

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "published": not errors,
            "expected_sources": expected_count,
            "sources": entries,
            "fragment_count": sum(entry.get("fragments", 0) for entry in entries),
            "errors": errors,
        }
        report = Path(staging) / "manifest.json"
        manifest["database_sha256"] = hashlib.sha256(database.read_bytes()).hexdigest() if not errors else None
        report.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if not errors:
            database.replace(output_dir / database.name)
        report.replace(output_dir / report.name)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--expected-count", type=int, default=21)
    args = parser.parse_args()
    report = build(args.raw_dir, args.output_dir, args.expected_count)
    print(f"TXT: {len(report['sources'])}. Фрагменты: {report['fragment_count']}. Ошибки: {len(report['errors'])}.")
    for error in report["errors"]:
        print(error)
    return 0 if report["published"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

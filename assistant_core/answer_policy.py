"""Политика ответа по результатам выбора редакции и проверки выводов."""
from __future__ import annotations

from datetime import date

from .contracts import (
    REFUSAL_UNLAWFUL,
    STATUS_CLARIFY,
    STATUS_FULL,
    STATUS_PARTIAL,
    STATUS_REFUSAL,
    Claim,
    Decision,
    Draft,
    QueryFacts,
    Selection,
)

UNLAWFUL_NOTICE = "Не могу помочь с нарушением закона или обходом обязательных требований."


def decide(
    facts: QueryFacts, selection: Selection, draft: Draft, verified: list[Claim]
) -> Decision:
    missing_fact = selection.missing_fact
    gaps = list(dict.fromkeys(gap.strip() for gap in selection.gaps if gap.strip()))
    cited_ids = {key for claim in verified for key in claim.fragment_ids}
    snapshots = {fragment.snapshot_as_of for fragment in selection.applicable
                 if fragment.fragment_id in cited_ids and fragment.snapshot_as_of}
    snapshot_notes = [
        f"Приведены положения из выгрузки от {date.fromisoformat(snapshot):%d.%m.%Y}. "
        "Применимость этих положений на запрошенную более позднюю дату не подтверждена."
        for snapshot in sorted(snapshots)
    ]
    if draft.refusal_reason == REFUSAL_UNLAWFUL or facts.unlawful_request:
        if verified:
            text = "\n".join(claim.text for claim in verified)
            note = " ".join([UNLAWFUL_NOTICE, *snapshot_notes])
            return Decision(
                STATUS_PARTIAL, f"{text}\n{note}",
                refusal_reason=UNLAWFUL_NOTICE, note=note,
            )
        return Decision(STATUS_REFUSAL, UNLAWFUL_NOTICE, refusal_reason=UNLAWFUL_NOTICE)
    gaps.extend(snapshot_notes)
    # Only echo parts of the user's question. A generated explanation of
    # a gap is not itself a verified legal statement.
    uncovered_parts = [
        part.strip() for part in draft.unanswered[:4]
        if 0 < len(part.strip()) <= 180 and part.strip() in facts.question
    ]
    gaps.extend(f"В базе не найдено достаточных оснований для части вопроса «{part}»"
                for part in dict.fromkeys(uncovered_parts))
    gap_text = " ".join(gap if gap.endswith((".", "?", "!")) else f"{gap}." for gap in gaps)
    question = f"Пожалуйста, укажите недостающие сведения ({missing_fact})." if missing_fact else ""

    if verified:
        text = "\n".join(claim.text for claim in verified)
        uncovered = draft.summary or len(verified) < len(draft.claims)
        if missing_fact or gap_text or uncovered:
            if uncovered and not (gap_text or missing_fact):
                gap_text = "Для остальных частей вопроса подтверждённых оснований недостаточно."
            note = "\n".join(part for part in (gap_text, question) if part)
            return Decision(STATUS_PARTIAL, f"{text}\n{note}", missing_fact=missing_fact, note=note)
        return Decision(STATUS_FULL, text)

    if missing_fact:
        text = " ".join(part for part in (gap_text, question) if part)
        return Decision(STATUS_CLARIFY, text, missing_fact=missing_fact)

    if draft.refusal_reason == "out_of_scope":
        reason = "Запрос выходит за рамки юридической помощи по выданной базе."
    else:
        reason = gap_text or "В выданной базе не найдено положений, подтверждающих ответ."
    return Decision(STATUS_REFUSAL, reason, refusal_reason=reason)

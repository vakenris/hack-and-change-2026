"""Три структурированных запроса к OpenAI-совместимому прокси."""
from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from calendar import monthrange
from dataclasses import asdict
from datetime import date

import httpx

from . import llm_prompts
from .contracts import Claim, Draft, EventDate, Fragment, Message, QueryFacts, TechnicalError
from .runtime import remaining

logger = logging.getLogger(__name__)


def _env(*names: str) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def _post(path: str, payload: dict, *, timeout: float, retries: int = 0) -> httpx.Response:
    base_url = _env("LLM_BASE_URL", "OPENAI_BASE_URL", "BASE_URL").rstrip("/")
    api_key = _env("RERANKER_API_KEY") if path == "/rerank" else ""
    api_key = api_key or _env("LLM_API_KEY", "OPENAI_API_KEY")
    if not base_url or not api_key:
        raise TechnicalError("Не заданы адрес или ключ прокси", code="proxy_config")

    for attempt in range(retries + 1):
        retry_after = 0.0
        try:
            response = httpx.post(
                f"{base_url}{path}",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
                timeout=min(timeout, remaining()),
            )
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            failure = TechnicalError(f"API модели вернул HTTP {status}", code=f"http_{status}")
            if status != 429 and not 500 <= status < 600:
                raise failure from None
            try:
                retry_after = max(0.0, float(exc.response.headers.get("Retry-After") or 0))
            except ValueError:
                pass
        except httpx.TimeoutException:
            failure = TechnicalError("Истекло время ожидания ответа модели", code="timeout")
        except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError):
            failure = TechnicalError("Ошибка соединения с API модели", code="connection")
        except (httpx.HTTPError, httpx.InvalidURL, ValueError):
            raise TechnicalError("Ошибка соединения с API модели", code="connection") from None

        if attempt == retries:
            raise failure from None
        delay = max(retry_after, 0.25 * 2 ** attempt)
        if delay >= min(remaining(), 5):
            raise failure from None
        time.sleep(delay)
    raise AssertionError("unreachable")


def _request(prompt: str, data: dict[str, object]) -> dict[str, object]:
    model = _env("LLM_MODEL", "OPENAI_MODEL", "GEN_MODEL")
    if not model:
        raise TechnicalError("Не задан идентификатор модели", code="proxy_config")

    try:
        timeout = float(_env("LLM_TIMEOUT") or "30")
        temperature = float(_env("TEMPERATURE") or "0")
        max_tokens = int(_env("MAX_TOKENS") or "4096")
        retries = int(_env("LLM_RETRIES") or "1")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout")
        if not 0 <= temperature <= 2 or max_tokens <= 0 or not 0 <= retries <= 2:
            raise ValueError("generation parameters")
    except ValueError:
        raise TechnicalError("Некорректные параметры модели в окружении", code="proxy_config") from None

    response = _post("/chat/completions", {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(data, ensure_ascii=False)},
        ],
    }, timeout=timeout, retries=retries)

    try:
        choice = response.json()["choices"][0]
        if choice.get("finish_reason") not in (None, "stop"):
            raise ValueError("finish_reason")
        content = choice["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("content")
        content = content.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", content, re.IGNORECASE | re.DOTALL)
        result = json.loads(fenced[1] if fenced else content)
        if not isinstance(result, dict):
            raise ValueError("result")
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        raise TechnicalError("API модели вернул некорректный структурированный ответ", code="model_format") from None
    return result


def _array(data: dict[str, object], field: str) -> list[object]:
    value = data.get(field)
    if not isinstance(value, list):
        raise TechnicalError(f"В ответе модели ожидается массив {field}")
    return value


def _string(value: object, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise TechnicalError(f"В ответе модели ожидается строка {field}")
    return value.strip()


def _strings(data: dict[str, object], field: str) -> list[str]:
    return list(dict.fromkeys(_string(item, field) for item in _array(data, field)))


def extract_facts(messages: list[Message]) -> QueryFacts:
    question = messages[-1].content
    result = _request(llm_prompts.EXTRACT_FACTS, {"messages": [asdict(m) for m in messages]})
    dates: list[EventDate] = []
    for item in _array(result, "dates"):
        if not isinstance(item, dict):
            raise TechnicalError("В ответе модели ожидается объект даты события")
        event_type = _string(item.get("event_type"), "event_type")
        value = _string(item.get("value"), "value")
        period = _string(item.get("period", ""), "period", allow_empty=True)
        try:
            if re.fullmatch(r"[0-9]{4}-[0-9]{2}", value):
                first = date.fromisoformat(value + "-01")
                last = first.replace(day=monthrange(first.year, first.month)[1])
                bounds = (first, last)
            elif re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
                bounds = (date.fromisoformat(value),)
            else:
                raise ValueError("date format")
        except ValueError:
            raise TechnicalError("Модель вернула некорректную дату события") from None
        for bound in bounds:
            event = EventDate(event_type, bound, period)
            if event not in dates:
                dates.append(event)
    return QueryFacts(
        question=question,
        search_queries=_strings(result, "search_queries") or [question],
        document_refs=_strings(result, "document_refs"),
        dates=dates,
    )


def _fragment_data(fragment: Fragment) -> dict[str, object]:
    data: dict[str, object] = {
        "fragment_id": fragment.fragment_id,
        "document_title": fragment.document_title,
        "revision_label": fragment.revision_label,
        "clause": fragment.clause,
        "text": fragment.text,
    }
    if fragment.periods:
        data["periods"] = fragment.periods
    return data


def draft_answer(
    messages: list[Message], facts: QueryFacts, fragments: list[Fragment]
) -> Draft:
    result = _request(
        llm_prompts.DRAFT_ANSWER,
        {
            "messages": [asdict(m) for m in messages],
            "facts": {
                "question": facts.question,
                "search_queries": facts.search_queries,
                "document_refs": facts.document_refs,
                "dates": [
                    {"event_type": event.event_type, "value": event.value.isoformat()}
                    for event in facts.dates
                ],
            },
            "fragments": [_fragment_data(fragment) for fragment in fragments],
        },
    )
    claims: list[Claim] = []
    invalid_claim: TechnicalError | None = None
    for item in _array(result, "claims"):
        try:
            if not isinstance(item, dict):
                raise TechnicalError("В ответе модели ожидается объект вывода")
            fragment_ids = _strings(item, "fragment_ids")
            if not fragment_ids:
                raise TechnicalError("Модель вернула вывод без ссылок на фрагменты")
            claims.append(Claim(_string(item.get("text"), "text"), fragment_ids))
        except TechnicalError as exc:
            invalid_claim = exc
    if invalid_claim is not None and not claims:
        raise invalid_claim
    reason = result.get("refusal_reason")
    if "refusal_reason" not in result or (
        reason is not None and (not isinstance(reason, str) or reason != "out_of_scope")
    ):
        raise TechnicalError("Модель вернула неизвестную причину отказа")
    summary = _string(result.get("summary"), "summary", allow_empty=True)
    if invalid_claim is not None and not summary:
        summary = "Не удалось подтвердить все части ответа"
    unanswered = _strings(result, "unanswered") if "unanswered" in result else []
    return Draft(claims, summary, reason, unanswered)


def verify_claims(
    claims: list[Claim], fragments: dict[str, Fragment], messages: list[Message],
) -> list[Claim]:
    if not claims:
        return []
    result = _request(
        llm_prompts.VERIFY_CLAIMS,
        {
            "user_context": [message.content for message in messages if message.role == "user"],
            "claims": [
                {
                    "index": index,
                    "text": claim.text,
                    "fragments": [_fragment_data(fragments[key]) for key in claim.fragment_ids],
                }
                for index, claim in enumerate(claims)
            ]
        },
    )
    indices = _array(result, "supported_indices")
    supported: set[int] = set()
    for index in indices:
        if type(index) is not int or not 0 <= index < len(claims):
            raise TechnicalError("Модель вернула неизвестный индекс вывода")
        supported.add(index)
    if len(supported) != len(indices):
        raise TechnicalError("Модель вернула повторяющийся индекс вывода")
    return [claim for index, claim in enumerate(claims) if index in supported]


def rerank_fragments(question: str, fragments: list[Fragment]) -> list[Fragment]:
    """Compare ranking on the same evidence set, preserving all selected sources."""
    model = _env("RERANKER_MODEL")
    if not model or len(fragments) < 2:
        return fragments
    try:
        top_n = min(len(fragments), max(1, int(_env("RERANKER_TOP_N") or len(fragments))))
        response = _post("/rerank", {
            "model": model,
            "query": question,
            "documents": [f"{item.document_title}, {item.clause}\n{item.text}" for item in fragments],
            "top_n": top_n,
        }, timeout=5.0)
        body = response.json()
        items = body.get("results", body.get("data"))
        if not isinstance(items, list) or not items:
            raise ValueError("empty ranking")
        indices = [item["index"] for item in items]
        if any(type(i) is not int or not 0 <= i < len(fragments) for i in indices):
            raise ValueError("unknown position")
        order = list(dict.fromkeys(indices + list(range(len(fragments)))))
        return [fragments[i] for i in order]
    except TechnicalError as error:
        if error.code == "deadline":
            raise
        logger.warning("Реранжирование недоступно: %s", error.code)
    except (AttributeError, KeyError, TypeError, ValueError):
        logger.warning("Реранжирование недоступно: model_format")
    return fragments

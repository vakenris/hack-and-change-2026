"""Три структурированных запроса к OpenAI-совместимому прокси."""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict
from datetime import date

import httpx

from . import llm_prompts
from .contracts import Claim, Draft, EventDate, Fragment, Message, QueryFacts, TechnicalError


def _env(*names: str) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def _request(prompt: str, data: dict) -> dict:
    base_url = _env("LLM_BASE_URL", "OPENAI_BASE_URL", "BASE_URL").rstrip("/")
    api_key = _env("LLM_API_KEY", "OPENAI_API_KEY")
    model = _env("LLM_MODEL", "OPENAI_MODEL", "GEN_MODEL")
    if not all((base_url, api_key, model)):
        raise TechnicalError("Не заданы адрес, ключ или идентификатор модели")

    try:
        timeout = float(_env("LLM_TIMEOUT") or "30")
        temperature = float(_env("TEMPERATURE") or "0")
        max_tokens = int(_env("MAX_TOKENS") or "4096")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout")
        if not 0 <= temperature <= 2 or max_tokens <= 0:
            raise ValueError("generation parameters")
    except ValueError:
        raise TechnicalError("Некорректные параметры модели в окружении") from None

    try:
        response = httpx.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(data, ensure_ascii=False)},
                ],
            },
            timeout=timeout,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise TechnicalError(f"API модели вернул HTTP {exc.response.status_code}") from None
    except httpx.TimeoutException:
        raise TechnicalError("Истекло время ожидания ответа модели") from None
    except (httpx.HTTPError, httpx.InvalidURL, ValueError):
        raise TechnicalError("Ошибка соединения с API модели") from None

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
        raise TechnicalError("API модели вернул некорректный структурированный ответ") from None
    return result


def _array(data: dict, field: str) -> list:
    value = data.get(field)
    if not isinstance(value, list):
        raise TechnicalError(f"В ответе модели ожидается массив {field}")
    return value


def _string(value: object, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise TechnicalError(f"В ответе модели ожидается строка {field}")
    return value.strip()


def _strings(data: dict, field: str) -> list[str]:
    return list(dict.fromkeys(_string(item, field) for item in _array(data, field)))


def extract_facts(messages: list[Message]) -> QueryFacts:
    question = messages[-1].content
    result = _request(llm_prompts.EXTRACT_FACTS, {"messages": [asdict(m) for m in messages]})
    dates = []
    for item in _array(result, "dates"):
        if not isinstance(item, dict):
            raise TechnicalError("В ответе модели ожидается объект даты события")
        event_type = _string(item.get("event_type"), "event_type")
        value = _string(item.get("value"), "value")
        try:
            if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
                raise ValueError("date format")
            event = EventDate(event_type, date.fromisoformat(value))
        except ValueError:
            raise TechnicalError("Модель вернула некорректную дату события") from None
        if event not in dates:
            dates.append(event)
    return QueryFacts(
        question=question,
        search_queries=_strings(result, "search_queries") or [question],
        document_refs=_strings(result, "document_refs"),
        dates=dates,
    )


def _fragment_data(fragment: Fragment) -> dict[str, str]:
    return {
        "fragment_id": fragment.fragment_id,
        "document_title": fragment.document_title,
        "revision_label": fragment.revision_label,
        "clause": fragment.clause,
        "text": fragment.text,
    }


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
    claims = []
    for item in _array(result, "claims"):
        if not isinstance(item, dict):
            raise TechnicalError("В ответе модели ожидается объект вывода")
        fragment_ids = _strings(item, "fragment_ids")
        if not fragment_ids:
            raise TechnicalError("Модель вернула вывод без ссылок на фрагменты")
        claims.append(Claim(_string(item.get("text"), "text"), fragment_ids))
    reason = result.get("refusal_reason")
    if "refusal_reason" not in result or reason not in (None, "out_of_scope"):
        raise TechnicalError("Модель вернула неизвестную причину отказа")
    return Draft(claims, _string(result.get("summary"), "summary", allow_empty=True), reason)


def verify_claims(claims: list[Claim], fragments: dict[str, Fragment]) -> list[Claim]:
    if not claims:
        return []
    result = _request(
        llm_prompts.VERIFY_CLAIMS,
        {
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
    if any(type(index) is not int or not 0 <= index < len(claims) for index in indices):
        raise TechnicalError("Модель вернула неизвестный индекс вывода")
    supported = set(indices)
    if len(supported) != len(indices):
        raise TechnicalError("Модель вернула повторяющийся индекс вывода")
    return [claim for index, claim in enumerate(claims) if index in supported]

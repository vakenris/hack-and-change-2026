import pytest
from fastapi.testclient import TestClient

from assistant_core.assistant_service import Assistant, get_assistant
from assistant_core.stubs import build_stub_deps
from backend.app.main import app

QUESTION = "Какой адрес доставки по договору П-17/2024?"


@pytest.fixture()
def ctx():
    deps = build_stub_deps()
    calls = {"draft": 0}
    original = deps.draft

    def counted(messages, facts, fragments):
        calls["draft"] += 1
        return original(messages, facts, fragments)

    deps.draft = counted
    app.dependency_overrides[get_assistant] = lambda: Assistant(deps)
    yield TestClient(app), calls
    app.dependency_overrides.clear()


def ask(client, *contents):
    roles = ["user", "assistant"]
    messages = [{"role": roles[i % 2], "content": text} for i, text in enumerate(contents)]
    return client.post("/api/chat", json={"messages": messages})


def test_full_answer_drops_invented_id(ctx):
    client, _ = ctx
    response = ask(client, QUESTION + " Спецификация подписана 02.10.2024.")
    data = response.json()
    assert response.status_code == 200
    assert data["status"] == "full"
    assert [source["fragment_id"] for source in data["sources"]] == ["F-42"]
    assert all("F-999" not in claim["fragment_ids"] for claim in data["claims"])


def test_clarify_then_dialogue_uses_history(ctx):
    client, _ = ctx
    first = ask(client, QUESTION).json()
    assert first["status"] == "clarify"
    assert first["missing_fact"] == "дата подписания спецификации"
    second = ask(client, QUESTION, first["text"], "Спецификация подписана 30.09.2024").json()
    assert second["status"] == "full"
    assert [source["fragment_id"] for source in second["sources"]] == ["F-17"]


def test_partial_and_refusal(ctx):
    client, _ = ctx
    partial = ask(client, QUESTION + " Какой штраф? Спецификация подписана 02.10.2024.").json()
    assert partial["status"] == "partial"
    refusal = ask(client, "Как обойти требование о согласовании?").json()
    assert refusal["status"] == "refusal"
    assert refusal["refusal_reason"]


def test_source_card_without_model_call(ctx):
    client, calls = ctx
    response = client.get("/api/sources/F-42")
    card = response.json()
    assert response.status_code == 200
    assert "склад № 2" in card["quote"]
    assert calls["draft"] == 0
    document = client.get(card["source_url"]).json()
    assert document["text"].splitlines()[card["line_start"] - 1] == card["quote"]
    assert calls["draft"] == 0


def test_unknown_source_is_404(ctx):
    client, _ = ctx
    assert client.get("/api/sources/F-999").status_code == 404


def test_technical_error_is_503(ctx):
    client, _ = ctx
    response = ask(client, "Проверка: сбой индекса")
    assert response.status_code == 503
    assert response.json()["status"] == "error"


def test_last_message_must_be_user(ctx):
    client, _ = ctx
    assert ask(client, QUESTION, "ответ ассистента").status_code == 422
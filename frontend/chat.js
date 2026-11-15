"use strict";

const sessionKey = "hackchange-legal-dialogue-v1";
const safeId = /^[A-Za-z0-9_-]+$/;
const labels = {
  partial: "Частичный ответ",
  clarify: "Нужно уточнение",
};

const byId = (id) => document.getElementById(id);
const ui = {
  composer: byId("composer"),
  question: byId("question"),
  send: byId("send"),
  newChat: byId("new-chat"),
  welcome: byId("welcome"),
  messages: byId("messages"),
  feedback: byId("chat-feedback"),
  error: byId("chat-error"),
  errorText: byId("chat-error-text"),
  retry: byId("retry"),
  panel: byId("sources-panel"),
  sourceFeedback: byId("source-feedback"),
  closeSource: byId("close-source"),
  card: byId("source-card"),
  title: byId("source-title"),
  meta: byId("source-meta"),
  quote: byId("source-quote"),
  openDocument: byId("open-document"),
  documentFeedback: byId("document-feedback"),
  documentView: byId("document-view"),
  documentLines: byId("document-lines"),
  documentLabel: byId("document-lines-label"),
};

let turns = loadTurns();
let chatRequest;
let sourceRequest;
let documentRequest;
let selectedSource;
let sourceTrigger;

function loadTurns() {
  try {
    const saved = JSON.parse(sessionStorage.getItem(sessionKey) || "[]");
    if (Array.isArray(saved)) {
      return saved.filter((turn) => turn && ["user", "assistant"].includes(turn.role)
        && typeof turn.content === "string");
    }
  } catch {
    // The dialogue remains usable if browser storage is unavailable.
  }
  return [];
}

function saveTurns() {
  try {
    sessionStorage.setItem(sessionKey, JSON.stringify(turns));
  } catch {
    // History is still held in memory for the current page.
  }
}

function create(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function sourceLabel(source) {
  const clause = source.clause
    ? (/^\d/.test(source.clause) ? `п. ${source.clause}` : source.clause)
    : "";
  return [source.document_title || "Источник", clause].filter(Boolean).join(", ");
}

function renderTurn(turn) {
  const item = create("li", `message ${turn.role}`);
  const text = create("p", "message-text");
  const speaker = turn.role === "user" ? "Вы" : "Ассистент";
  text.append(create("span", "visually-hidden", `${speaker}. `));
  if (turn.role === "assistant" && Object.hasOwn(labels, turn.status)) {
    text.classList.add("needs-attention");
    text.append(create("span", "visually-hidden", `${labels[turn.status]}. `));
  }
  text.append(turn.content);
  item.append(text);
  if (Array.isArray(turn.sources) && turn.sources.length) {
    const list = create("ul", "answer-sources");
    list.setAttribute("aria-label", "Источники ответа");
    for (const source of turn.sources) {
      if (!source || !safeId.test(source.fragment_id || "")) continue;
      const entry = create("li");
      const button = create("button", "answer-source", sourceLabel(source));
      button.type = "button";
      button.dataset.fragmentId = source.fragment_id;
      button.setAttribute("aria-controls", "sources-panel");
      button.setAttribute("aria-expanded", "false");
      button.addEventListener("click", () => openSource(source.fragment_id, button));
      entry.append(button);
      list.append(entry);
    }
    if (list.childElementCount) item.append(list);
  }
  return item;
}

function renderTurns() {
  ui.welcome.hidden = turns.length > 0;
  ui.messages.replaceChildren(...turns.map(renderTurn));
}

function setBusy(busy) {
  ui.send.disabled = busy;
  ui.retry.disabled = busy;
  ui.question.readOnly = busy;
  ui.composer.setAttribute("aria-busy", String(busy));
  ui.feedback.textContent = busy ? "Ищу подходящие положения…" : "";
}

function showChatError(text) {
  ui.error.hidden = !text;
  ui.errorText.textContent = text;
}

async function requestJson(path, options) {
  const response = await fetch(path, options);
  let payload;
  try {
    payload = await response.json();
  } catch {
    throw new Error("Сервер вернул ответ в неизвестном формате.");
  }
  if (!response.ok) {
    if (response.status === 404) throw new Error("Источник не найден в базе.");
    if (response.status === 503) throw new Error("Сервис временно недоступен.");
    if (response.status === 422) throw new Error("Сервер не смог обработать запрос.");
    throw new Error("Не удалось выполнить запрос.");
  }
  return payload;
}

function appendTurn(turn) {
  turns.push(turn);
  ui.welcome.hidden = true;
  const item = renderTurn(turn);
  ui.messages.append(item);
  saveTurns();
  item.scrollIntoView({ block: "nearest" });
}

async function sendQuestion(retry = false) {
  if (chatRequest) return;
  const content = retry ? "" : ui.question.value.trim();
  if (content) {
    if (turns.at(-1)?.role === "user") {
      turns.pop();
      ui.messages.lastElementChild?.remove();
    }
    appendTurn({ role: "user", content });
    ui.question.value = "";
  } else if (turns.at(-1)?.role !== "user") {
    return;
  }
  const controller = new AbortController();
  chatRequest = controller;
  setBusy(true);
  showChatError("");
  try {
    const messages = turns.slice(-49).map(({ role, content }) => ({ role, content }));
    const answer = await requestJson("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages }),
      signal: controller.signal,
    });
    if (chatRequest !== controller) return;
    if (!answer || typeof answer.text !== "string" || !answer.text.trim() || answer.status === "error") {
      throw new Error("Не удалось получить ответ ассистента.");
    }
    appendTurn({
      role: "assistant",
      content: answer.text,
      status: answer.status,
      sources: answer.sources,
    });
  } catch (error) {
    if (controller.signal.aborted) return;
    showChatError(error instanceof TypeError
      ? "Связь с сервером прервалась. Проверьте подключение."
      : error.message);
  } finally {
    if (chatRequest === controller) {
      chatRequest = undefined;
      setBusy(false);
      if (!ui.error.hidden) ui.retry.focus();
      else if (document.activeElement === document.body) ui.question.focus({ preventScroll: true });
    }
  }
}

function setFeedback(element, text) {
  element.hidden = !text;
  element.textContent = text;
}

function clearDocument() {
  documentRequest?.abort();
  documentRequest = undefined;
  ui.documentView.hidden = true;
  ui.documentLines.replaceChildren();
  ui.openDocument.disabled = false;
  ui.openDocument.textContent = "Открыть документ";
  ui.openDocument.setAttribute("aria-expanded", "false");
  setFeedback(ui.documentFeedback, "");
}

function closeSource(restoreFocus = true) {
  sourceRequest?.abort();
  sourceRequest = undefined;
  clearDocument();
  selectedSource = undefined;
  ui.card.hidden = true;
  ui.panel.hidden = true;
  setFeedback(ui.sourceFeedback, "");
  if (sourceTrigger) {
    sourceTrigger.setAttribute("aria-expanded", "false");
    if (restoreFocus) sourceTrigger.focus();
  }
  sourceTrigger = undefined;
}

async function openSource(fragmentId, trigger) {
  closeSource(false);
  sourceTrigger = trigger;
  trigger.setAttribute("aria-expanded", "true");
  ui.panel.hidden = false;
  setFeedback(ui.sourceFeedback, "Загружаю источник…");
  ui.panel.focus({ preventScroll: true });
  ui.panel.scrollIntoView({ block: "nearest" });
  const controller = new AbortController();
  sourceRequest = controller;
  try {
    const card = await requestJson(`/api/sources/${encodeURIComponent(fragmentId)}`, { signal: controller.signal });
    if (sourceRequest !== controller) return;
    if (!card || typeof card.quote !== "string" || !safeId.test(card.document_id || "")) {
      throw new Error("Сервер вернул неполную карточку. Откройте источник повторно.");
    }
    selectedSource = card;
    ui.title.textContent = card.document_title || "Источник";
    const clause = card.clause && /^\d/.test(card.clause) ? `п. ${card.clause}` : card.clause;
    const meta = [clause, card.revision_label].filter(Boolean).join(" · ");
    ui.meta.textContent = meta;
    ui.meta.hidden = !meta;
    ui.quote.textContent = card.quote;
    setFeedback(ui.sourceFeedback, "");
    ui.card.hidden = false;
  } catch (error) {
    if (!controller.signal.aborted) {
      setFeedback(ui.sourceFeedback, error instanceof TypeError
        ? "Не удалось загрузить источник. Нажмите на него ещё раз." : error.message);
    }
  }
}

function renderDocument(documentData, card) {
  if (!documentData || typeof documentData.text !== "string") {
    throw new Error("Сервер вернул документ без текста.");
  }
  const lines = documentData.text.split(/\r\n|\n|\r/);
  const start = Number(card.line_start);
  const end = Number(card.line_end);
  const validRange = Number.isInteger(start) && Number.isInteger(end)
    && start >= 1 && end >= start && end <= lines.length;
  const list = document.createDocumentFragment();
  for (const [index, text] of lines.entries()) {
    const number = index + 1;
    const entry = create("li", validRange && number >= start && number <= end ? "is-quoted" : "");
    const label = create("span", "line-number", String(number));
    label.setAttribute("aria-hidden", "true");
    entry.append(label, create("span", "line-text", text));
    list.append(entry);
  }
  ui.documentLabel.textContent = validRange
    ? (start === end ? `Цитата выделена в строке ${start}` : `Цитата выделена в строках ${start}–${end}`)
    : "Полный текст документа";
  ui.documentLines.replaceChildren(list);
  ui.documentView.hidden = false;
  ui.openDocument.setAttribute("aria-expanded", "true");
  ui.openDocument.textContent = "Скрыть документ";
  ui.documentView.scrollIntoView({ block: "nearest" });
  const selected = ui.documentLines.querySelector(".is-quoted");
  ui.documentLines.focus({ preventScroll: true });
  if (selected) {
    const container = ui.documentLines.getBoundingClientRect();
    ui.documentLines.scrollTop += selected.getBoundingClientRect().top - container.top - container.height / 3;
  }
}

async function toggleDocument() {
  if (!ui.documentView.hidden) {
    clearDocument();
    return;
  }
  const card = selectedSource;
  if (!card || documentRequest) return;
  const controller = new AbortController();
  documentRequest = controller;
  ui.openDocument.disabled = true;
  setFeedback(ui.documentFeedback, "Загружаю документ…");
  const params = new URLSearchParams({ line_start: card.line_start, line_end: card.line_end });
  try {
    const data = await requestJson(`/api/documents/${encodeURIComponent(card.document_id)}?${params}`,
      { signal: controller.signal });
    if (documentRequest !== controller) return;
    renderDocument(data, card);
    setFeedback(ui.documentFeedback, "");
  } catch (error) {
    if (!controller.signal.aborted) {
      setFeedback(ui.documentFeedback, error instanceof TypeError
        ? "Не удалось загрузить документ. Попробуйте ещё раз." : error.message);
    }
  } finally {
    if (documentRequest === controller) {
      documentRequest = undefined;
      ui.openDocument.disabled = false;
    }
  }
}

ui.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  sendQuestion();
});
ui.question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    ui.composer.requestSubmit();
  }
});
ui.retry.addEventListener("click", () => sendQuestion(true));
ui.newChat.addEventListener("click", () => {
  chatRequest?.abort();
  chatRequest = undefined;
  closeSource(false);
  turns = [];
  saveTurns();
  renderTurns();
  setBusy(false);
  showChatError("");
  ui.question.value = "";
  ui.question.focus();
});
ui.closeSource.addEventListener("click", () => closeSource());
ui.panel.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeSource();
});
ui.openDocument.addEventListener("click", toggleDocument);
ui.openDocument.setAttribute("aria-controls", "document-view");
ui.openDocument.setAttribute("aria-expanded", "false");
for (const button of ui.welcome.querySelectorAll("[data-prompt]")) {
  button.addEventListener("click", () => {
    ui.question.value = button.dataset.prompt;
    ui.question.focus();
  });
}

renderTurns();
if (turns.at(-1)?.role === "user") showChatError("Последний вопрос остался без ответа.");

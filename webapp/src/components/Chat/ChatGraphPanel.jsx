import React, { useEffect, useMemo, useRef } from "react";

const PROMPT_LEAK_MARKERS = [
  "behavioral_dialogue_simulation",
  "behavioral_dialogue_simulation_fast",
  "\"task\": \"behavioral_dialogue_simulation",
  "\"task\": \"behavioral_dialogue_simulation_fast",
  "use only the ram graph and personality data below",
  "do not answer like a polite assistant",
  "return plain text only",
  "do not output json",
  "dialogue_contract",
  "ram_context",
  "agent_plan",
  "system instruction",
];

function ThreadMessage({ item, t }) {
  const role = item.role === "user" ? "user" : "assistant";
  return (
    <article className={`chat-thread-item ${role}`}>
      <header>
        <strong>{role === "user" ? t("chat_you") : t("chat_assistant")}</strong>
        <span>{item.timestamp}</span>
      </header>
      <p>{item.message}</p>
    </article>
  );
}

export function ChatGraphPanel({
  session,
  value,
  onChange,
  running,
  progress,
  onRun,
  t,
}) {
  const threadRef = useRef(null);
  const messages = useMemo(() => listMessages(session), [session]);

  useEffect(() => {
    const node = threadRef.current;
    if (!node) return;
    node.scrollTop = node.scrollHeight;
  }, [messages.length, running]);

  const onSubmit = (event) => {
    event.preventDefault();
    if (!value.trim() || running) return;
    onRun();
  };

  return (
    <section className="chat-workspace-panel glass-panel">
      <header className="panel-heading compact">
        <div>
          <p className="eyebrow">{t("chat_eyebrow")}</p>
          <h2>{session?.title || t("chat_title")}</h2>
        </div>
        {progress ? <span className="chat-progress-label">{progress}</span> : null}
      </header>

      <div ref={threadRef} className="chat-thread-scroll">
        {!messages.length ? (
          <div className="empty-state large">
            <h3>{t("chat_empty_title")}</h3>
            <p>{t("chat_empty_text")}</p>
          </div>
        ) : (
          <div className="chat-thread">
            {messages.map((item) => <ThreadMessage key={item.id} item={item} t={t} />)}
          </div>
        )}
      </div>

      <form className="chat-composer-fixed" onSubmit={onSubmit}>
        <label className="field-stack">
          <span>{t("chat_message")}</span>
          <textarea
            value={value}
            onChange={(event) => onChange(event.target.value)}
            placeholder={t("chat_message_placeholder")}
          />
        </label>
        <div className="chat-composer-actions">
          <button type="submit" disabled={running || !value.trim()}>
            {running ? t("chat_running") : t("chat_send")}
          </button>
        </div>
      </form>
    </section>
  );
}

function listMessages(session) {
  if (!session || !Array.isArray(session.messages)) {
    return [];
  }
  return session.messages
    .map((item, index) => ({
      id: item.id || `${item.role || "msg"}-${index}`,
      role: item.role || "assistant",
      message: normalizeAssistantMessage(item.role || "assistant", item.message || ""),
      timestamp: item.timestamp || "",
    }))
    .filter((item) => item.role === "user" || item.message.trim());
}

function normalizeAssistantMessage(role, message) {
  const raw = String(message || "");
  if (role === "user") {
    return raw;
  }
  if (looksLikePromptLeak(raw)) {
    return "";
  }
  const trimmed = raw.trim();
  const candidates = [];
  if (trimmed.startsWith("{") && trimmed.endsWith("}")) {
    candidates.push(trimmed);
  }
  const start = trimmed.indexOf("{");
  const end = trimmed.lastIndexOf("}");
  if (start !== -1 && end > start) {
    candidates.push(trimmed.slice(start, end + 1));
  }
  try {
    for (const candidate of candidates) {
      const parsed = JSON.parse(candidate);
      if (parsed && typeof parsed === "object") {
        const keys = ["assistant_reply", "reply", "text", "message", "content", "response"];
        for (const key of keys) {
          const value = parsed[key];
          if (typeof value === "string" && value.trim()) {
            return looksLikePromptLeak(value) ? "" : value.trim();
          }
        }
        const character = String(parsed.character || parsed.name || "").trim();
        const traits = Array.isArray(parsed.traits)
          ? parsed.traits.map((item) => String(item || "").trim()).filter(Boolean)
          : [];
        if (character && traits.length) {
          return `${character}: ${traits.join("; ")}`;
        }
      }
    }
  } catch (_) {
    return raw;
  }
  return looksLikePromptLeak(raw) ? "" : raw;
}

function looksLikePromptLeak(value) {
  const lowered = String(value || "").trim().toLowerCase();
  if (!lowered) {
    return false;
  }
  return PROMPT_LEAK_MARKERS.some((marker) => lowered.includes(marker));
}

import React, { useEffect, useMemo, useRef, useState } from "react";

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
  "analyze the request",
  "# анализ ситуации",
  "ключевая проблема",
  "safety/policy",
  "output format:",
  "что я могу сказать",
  "что я не могу",
  "внешний ответ персонажа",
  "review notes",
  "issues identified",
  "thinking process:",
  "<think>",
  "</think>",
];

const HIDDEN_DEFAULTS = new Set([
  "absent",
  "neutral",
  "unclear",
  "continuation",
  "defined",
  "direct",
  "literal",
  "statement",
  "non_answer",
  "independent",
  "opening_new_direction",
  "calm",
]);

const GENERIC_ASSISTANT_NAMES = new Set([
  "assistant",
  "ассистент",
  "օգնական",
  "助手",
  "ai assistant",
  "default assistant",
]);

const DEFAULT_ASSISTANT_SPEAKER = "LLM";

function isGenericAssistantLabel(value) {
  const normalized = String(value || "").trim().toLowerCase();
  return Boolean(normalized) && GENERIC_ASSISTANT_NAMES.has(normalized);
}

function resolveAssistantSpeakerName(item, activePersonaName) {
  if (item?.role === "user") {
    return "";
  }
  const storedPersonaName = String(item?.persona_name || "").trim();
  const storedSpeakerName = String(item?.speaker_name || "").trim();
  const explicitPersonaName = String(activePersonaName || "").trim();
  // Prefer the persona name stored in the message record — it's accurate for that message.
  // Fall back to the currently selected persona only when no record exists (legacy messages).
  const resolvedSpeakerName = storedPersonaName
    || explicitPersonaName
    || (!isGenericAssistantLabel(storedSpeakerName) ? storedSpeakerName : "")
    || DEFAULT_ASSISTANT_SPEAKER;
  return `${resolvedSpeakerName}`;
}

function ThreadMessage({ item, t, trainingMode, activePersonaName, userPersonaName, selected, onSelect }) {
  const role = item.role === "user" ? "user" : "assistant";
  const preview = summarizeVector(item.vector);

  return (
    <article
      className={[
        "chat-thread-item",
        role,
        item.pending ? "pending" : "",
        trainingMode ? "training-target" : "",
        selected ? "selected" : "",
      ].filter(Boolean).join(" ")}
      onClick={trainingMode ? () => onSelect(item.message_id) : undefined}
      role={trainingMode ? "button" : undefined}
      tabIndex={trainingMode ? 0 : undefined}
      onKeyDown={trainingMode ? (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onSelect(item.message_id);
        }
      } : undefined}
    >
      <header>
        <div className="chat-thread-item__identity">
          <strong>{role === "user" ? (item.user_persona_name || userPersonaName || t("chat_user_persona_stranger")) : resolveAssistantSpeakerName(item, activePersonaName)}</strong>
          {trainingMode ? (
            <span className={`training-label ${item.has_correction ? "saved" : ""}`}>
              {item.has_correction ? t("annotation_corrected") : t("annotation_pending")}
            </span>
          ) : null}
        </div>
        <span>{item.timestamp}</span>
      </header>
      <p>{item.message}</p>
      {preview ? <p className="annotation-inline-summary">{preview}</p> : null}
    </article>
  );
}

function isDefaultValue(v) {
  return !v || HIDDEN_DEFAULTS.has(v);
}

function CoordinateRow({ spec, choice, predicted, onChange, onToggleExtra }) {
  const extras = Array.isArray(choice?.extra) ? choice.extra : [];
  const predictedMain = predicted?.main || "";
  const currentMain = choice?.main || spec.options?.[0] || "";
  const isDirty = currentMain !== (predicted?.main || spec.options?.[0] || "");
  return (
    <div className={`annot-coord-row ${isDirty ? "dirty" : ""}`}>
      <div className="annot-coord-row__id" title={spec.description}>
        <strong>{spec.interpreter_id}</strong>
        <span>{spec.title}</span>
      </div>
      <select
        className="annot-coord-row__select"
        value={currentMain}
        onChange={(e) => onChange(spec.interpreter_id, e.target.value)}
      >
        {(spec.options || []).map((opt) => (
          <option key={opt} value={opt}>{opt}</option>
        ))}
      </select>
      {predictedMain && predictedMain !== currentMain && (
        <button
          type="button"
          className="annot-coord-row__predicted"
          title={`Predicted: ${predictedMain}`}
          onClick={() => onChange(spec.interpreter_id, predictedMain)}
        >
          ← {predictedMain}
        </button>
      )}
      <div className="annot-coord-row__extras">
        {(spec.options || []).filter((o) => o !== currentMain).map((opt) => (
          <label key={opt} className={`annot-extra-chip ${extras.includes(opt) ? "on" : ""}`}>
            <input type="checkbox" checked={extras.includes(opt)} onChange={() => onToggleExtra(spec.interpreter_id, opt)} />
            {opt}
          </label>
        ))}
      </div>
    </div>
  );
}

function MessageAnnotationEditor({
  selectedMessage,
  registry,
  messagesById,
  onSaveAnnotation,
  onSaveCorrection,
  onSelectMessage,
  activePersonaName,
  userPersonaName,
  t,
}) {
  const [draftVector, setDraftVector] = useState({});
  const [notes, setNotes] = useState("");
  const [correctOutput, setCorrectOutput] = useState("");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [correctionSaved, setCorrectionSaved] = useState(false);
  const [openGroups, setOpenGroups] = useState([]);
  const [filterText, setFilterText] = useState("");

  useEffect(() => {
    setDraftVector(cloneVector(selectedMessage?.vector || {}));
    setNotes(selectedMessage?.annotation_notes || "");
    setCorrectOutput("");
    setSaved(false);
    setCorrectionSaved(false);
    setFilterText("");
    // Auto-open groups that have non-default predicted values
    const predicted = selectedMessage?.predicted_vector || selectedMessage?.vector || {};
    const activeGroups = new Set();
    for (const spec of registry || []) {
      const main = (predicted[spec.interpreter_id] || {}).main || "";
      if (!isDefaultValue(main)) activeGroups.add(spec.group_id);
    }
    setOpenGroups(Array.from(activeGroups));
  }, [selectedMessage?.message_id, selectedMessage?.annotation_id]);

  const groupedRegistry = useMemo(() => {
    const groups = new Map();
    for (const spec of registry || []) {
      const groupId = spec.group_id || "misc";
      if (!groups.has(groupId)) {
        groups.set(groupId, { groupId, title: spec.group_title || groupId, items: [] });
      }
      groups.get(groupId).items.push(spec);
    }
    return Array.from(groups.values());
  }, [registry]);

  // Signals = P-coords with non-default value in draft or predicted
  const activeSignals = useMemo(() => {
    const predicted = selectedMessage?.predicted_vector || {};
    return (registry || []).filter((spec) => {
      const draftMain = (draftVector[spec.interpreter_id] || {}).main || "";
      const predMain = (predicted[spec.interpreter_id] || {}).main || "";
      return !isDefaultValue(draftMain) || !isDefaultValue(predMain);
    });
  }, [registry, draftVector, selectedMessage?.predicted_vector]);

  const filteredGroups = useMemo(() => {
    if (!filterText.trim()) return groupedRegistry;
    const q = filterText.trim().toLowerCase();
    return groupedRegistry
      .map((g) => ({
        ...g,
        items: g.items.filter(
          (s) => s.interpreter_id.toLowerCase().includes(q) || s.title.toLowerCase().includes(q)
        ),
      }))
      .filter((g) => g.items.length > 0);
  }, [groupedRegistry, filterText]);

  if (!selectedMessage) {
    return (
      <section className="annotation-workbench">
        <header className="annotation-workbench__header">
          <div>
            <p className="eyebrow">{t("annotation_title")}</p>
            <h3>{t("annotation_select_message")}</h3>
          </div>
        </header>
      </section>
    );
  }

  function updateMain(interpreterId, nextMain) {
    setDraftVector((current) => {
      const next = cloneVector(current);
      const existing = next[interpreterId] || { main: nextMain, extra: [] };
      next[interpreterId] = {
        main: nextMain,
        extra: Array.isArray(existing.extra) ? existing.extra.filter((item) => item !== nextMain) : [],
      };
      return next;
    });
    setSaved(false);
  }

  function toggleExtra(interpreterId, label) {
    setDraftVector((current) => {
      const next = cloneVector(current);
      const existing = next[interpreterId] || { main: "", extra: [] };
      const extras = Array.isArray(existing.extra) ? [...existing.extra] : [];
      const index = extras.indexOf(label);
      if (index === -1) {
        extras.push(label);
      } else {
        extras.splice(index, 1);
      }
      next[interpreterId] = {
        ...existing,
        extra: extras.filter((item) => item && item !== existing.main),
      };
      return next;
    });
    setSaved(false);
  }

  function toggleGroup(groupId) {
    setOpenGroups((current) => current.includes(groupId)
      ? current.filter((item) => item !== groupId)
      : [...current, groupId]);
  }

  function acceptPredicted() {
    const predicted = selectedMessage?.predicted_vector || {};
    setDraftVector(cloneVector(Object.keys(predicted).length ? predicted : selectedMessage?.vector || {}));
    setSaved(false);
  }

  async function handleSave() {
    if (saving) return;
    setSaving(true);
    try {
      const [annotResult, corrResult] = await Promise.all([
        onSaveAnnotation({
          message_id: selectedMessage.message_id,
          role: selectedMessage.role,
          raw_text: selectedMessage.raw_text,
          analysis_text: selectedMessage.analysis_text,
          display_text: selectedMessage.display_text,
          timestamp: selectedMessage.timestamp,
          persona_name: selectedMessage.persona_name,
          coordinates: draftVector,
          context_window: selectedMessage.context_window || [],
          context_matrix: selectedMessage.context_matrix || [],
          transition_interpretation: selectedMessage.transition_interpretation || {},
          notes,
        }),
        correctOutput.trim() && onSaveCorrection
          ? onSaveCorrection({
              input_text: selectedMessage.raw_text || selectedMessage.display_text || "",
              correct_output: correctOutput.trim(),
              persona_name: selectedMessage.persona_name || "",
            })
          : Promise.resolve(null),
      ]);
      setSaved(Boolean(annotResult));
      if (corrResult) setCorrectionSaved(true);
    } finally {
      setSaving(false);
    }
  }

  return (
    <section className="annotation-workbench">
      <header className="annotation-workbench__header">
        <div>
          <p className="eyebrow">{t("annotation_title")}</p>
          <strong>
            {selectedMessage.role === "user"
              ? (selectedMessage.user_persona_name || t("chat_user_persona_stranger"))
              : resolveAssistantSpeakerName(selectedMessage, activePersonaName)}
          </strong>
          <span className="annot-msg-preview">{selectedMessage.display_text?.slice(0, 80)}{selectedMessage.display_text?.length > 80 ? "…" : ""}</span>
        </div>
        <div className="annotation-workbench__actions">
          {saved ? <span className="training-saved-badge">{t("training_saved")}</span> : null}
          <button type="button" className="button-secondary" title="Accept all predicted values" onClick={acceptPredicted}>
            ↺ predicted
          </button>
          <button type="button" className="button-primary" disabled={saving} onClick={handleSave}>
            {saving ? "…" : t("annotation_save")}
          </button>
        </div>
      </header>

      {/* Active signals — only non-default P-coords */}
      {activeSignals.length > 0 && (
        <section className="annot-signals">
          <p className="annot-signals__label">Активные сигналы</p>
          {activeSignals.map((spec) => (
            <CoordinateRow
              key={spec.interpreter_id}
              spec={spec}
              choice={draftVector?.[spec.interpreter_id] || { main: spec.options?.[0] || "", extra: [] }}
              predicted={selectedMessage.predicted_vector?.[spec.interpreter_id] || {}}
              onChange={updateMain}
              onToggleExtra={toggleExtra}
            />
          ))}
        </section>
      )}

      {/* Search + full registry */}
      <section className="annot-all">
        <div className="annot-search-row">
          <input
            type="search"
            className="annot-search"
            placeholder="Поиск P-координаты…"
            value={filterText}
            onChange={(e) => setFilterText(e.target.value)}
          />
          <button type="button" className="link-button" onClick={() => setOpenGroups(filteredGroups.map((g) => g.groupId))}>
            развернуть все
          </button>
          <button type="button" className="link-button" onClick={() => setOpenGroups([])}>
            свернуть все
          </button>
        </div>

        {filteredGroups.map((group) => {
          const isOpen = openGroups.includes(group.groupId) || Boolean(filterText.trim());
          const nonDefaultCount = group.items.filter((s) => {
            const m = (draftVector?.[s.interpreter_id] || {}).main || "";
            return !isDefaultValue(m);
          }).length;
          return (
            <section key={group.groupId} className="annot-group">
              <button type="button" className="annot-group__toggle" onClick={() => toggleGroup(group.groupId)}>
                <strong>{group.groupId}</strong>
                <span>{group.title}</span>
                {nonDefaultCount > 0 && <span className="annot-group__badge">{nonDefaultCount}</span>}
                <small>{isOpen ? "−" : "+"}</small>
              </button>
              {isOpen && group.items.map((spec) => (
                <CoordinateRow
                  key={spec.interpreter_id}
                  spec={spec}
                  choice={draftVector?.[spec.interpreter_id] || { main: spec.options?.[0] || "", extra: [] }}
                  predicted={selectedMessage.predicted_vector?.[spec.interpreter_id] || {}}
                  onChange={updateMain}
                  onToggleExtra={toggleExtra}
                />
              ))}
            </section>
          );
        })}
      </section>

      <div className="annot-bottom">
        <label className="field-stack annot-correction">
          <span className="annot-correction__label">{t("annotation_correct_response")}</span>
          <textarea
            className="annot-correction__input"
            value={correctOutput}
            onChange={(e) => { setCorrectOutput(e.target.value); setCorrectionSaved(false); }}
            rows={3}
            placeholder={t("annotation_correct_placeholder")}
          />
        </label>
        <label className="field-stack annotation-notes">
          <span>{t("annotation_notes")}</span>
          <textarea
            value={notes}
            onChange={(e) => { setNotes(e.target.value); setSaved(false); }}
            rows={2}
            placeholder={t("annotation_notes_placeholder")}
          />
        </label>
        <div className="annot-bottom__actions">
          {correctionSaved ? <span className="training-saved-badge correction">{t("annotation_correction_saved")}</span> : null}
          {saved ? <span className="training-saved-badge">{t("training_saved")}</span> : null}
          <button type="button" className="button-primary" disabled={saving} onClick={handleSave}>
            {saving ? "…" : t("annotation_save")}
          </button>
        </div>
      </div>
    </section>
  );
}

export function ChatGraphPanel({
  session,
  annotationWorkspace,
  optimisticMessages,
  value,
  onChange,
  running,
  progress,
  composerResetToken,
  onRun,
  trainingMode,
  onSaveAnnotation,
  onSaveCorrection,
  activePersonaName,
  personas,
  userPersonaName,
  onUserPersonaChange,
  t,
}) {
  const threadRef = useRef(null);
  const messages = useMemo(
    () => listMessages(session, annotationWorkspace, optimisticMessages),
    [session, annotationWorkspace, optimisticMessages],
  );
  const resolvedActivePersonaName = useMemo(() => {
    const explicitName = String(activePersonaName || "").trim();
    if (explicitName) {
      return explicitName;
    }
    const fallbackMessage = [...messages].reverse().find((item) => item.role !== "user" && String(item.persona_name || "").trim());
    return String(fallbackMessage?.persona_name || "").trim();
  }, [activePersonaName, messages]);
  const messagesById = useMemo(() => new Map(messages.map((item) => [item.message_id, item])), [messages]);
  const registry = annotationWorkspace?.registry || [];
  const [selectedMessageId, setSelectedMessageId] = useState("");

  useEffect(() => {
    const node = threadRef.current;
    if (!node) return;
    node.scrollTop = node.scrollHeight;
  }, [messages.length, running]);

  useEffect(() => {
    if (!messages.length) {
      setSelectedMessageId("");
      return;
    }
    if (!selectedMessageId || !messagesById.has(selectedMessageId)) {
      setSelectedMessageId(messages[messages.length - 1].message_id);
    }
  }, [messages, messagesById, selectedMessageId]);

  const selectedMessage = useMemo(
    () => messages.find((item) => item.message_id === selectedMessageId) || messages[messages.length - 1] || null,
    [messages, selectedMessageId],
  );

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
            {messages.map((item) => (
              <ThreadMessage
                key={item.message_id}
                item={item}
                t={t}
                trainingMode={trainingMode}
                activePersonaName={resolvedActivePersonaName}
                userPersonaName={userPersonaName || ""}
                selected={trainingMode && item.message_id === selectedMessageId}
                onSelect={setSelectedMessageId}
              />
            ))}
          </div>
        )}
      </div>

      {trainingMode ? (
        <MessageAnnotationEditor
          selectedMessage={selectedMessage}
          registry={registry}
          messagesById={messagesById}
          onSaveAnnotation={onSaveAnnotation}
          onSaveCorrection={onSaveCorrection}
          onSelectMessage={setSelectedMessageId}
          activePersonaName={resolvedActivePersonaName}
          userPersonaName={userPersonaName || ""}
          t={t}
        />
      ) : null}

      <form className="chat-composer-fixed" onSubmit={onSubmit}>
        <div className="chat-composer-persona-row">
          <label className="chat-composer-persona-label">
            <span>{t("chat_user_persona_label")}</span>
            <select
              className="chat-composer-persona-select"
              value={userPersonaName || ""}
              onChange={(event) => onUserPersonaChange && onUserPersonaChange(event.target.value)}
            >
              <option value="">{t("chat_user_persona_stranger")}</option>
              {(personas || []).map((item) => (
                <option key={item.name} value={item.name}>{item.label || item.profile?.name || item.name}</option>
              ))}
            </select>
          </label>
        </div>
        <label className="field-stack">
          <textarea
            key={`chat-composer-${composerResetToken || 0}`}
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

function listMessages(session, annotationWorkspace, optimisticMessages = []) {
  const annotationRows = Array.isArray(annotationWorkspace?.messages) ? annotationWorkspace.messages : null;
  const sourceRows = annotationRows || (Array.isArray(session?.messages) ? session.messages : []);
  const normalizedRows = sourceRows
    .map((item, index) => {
      const role = item.role || "assistant";
      const displayText = item.display_text || item.raw_text || item.message || "";
      return {
        id: item.message_id || item.id || `${role}-${index}`,
        message_id: item.message_id || item.id || `${role}-${index}`,
        role,
        message: normalizeAssistantMessage(role, displayText),
        raw_text: item.raw_text || displayText,
        display_text: displayText,
        analysis_text: item.analysis_text || displayText,
        timestamp: item.timestamp || "",
        persona_name: item.persona_name || "",
        user_persona_name: item.user_persona_name || "",
        speaker_name: item.speaker_name || "",
        vector: item.vector || {},
        predicted_vector: item.predicted_vector || {},
        context_matrix: item.context_matrix || [],
        context_window: item.context_window || [],
        history_flow: item.history_flow || [],
        transition_interpretation: item.transition_interpretation || {},
        has_correction: Boolean(item.has_correction),
        annotation_notes: item.annotation_notes || "",
        annotation_id: item.annotation_id || "",
        pending: Boolean(item.pending),
      };
    })
    .filter((item) => item.role === "user" || item.message.trim());
  const normalizedOptimisticRows = Array.isArray(optimisticMessages)
    ? optimisticMessages.map((item, index) => {
      const role = item.role || "assistant";
      const displayText = item.display_text || item.raw_text || item.message || "";
      return {
        id: item.message_id || item.id || `optimistic-${role}-${index}`,
        message_id: item.message_id || item.id || `optimistic-${role}-${index}`,
        role,
        message: role === "user" ? String(displayText || "") : normalizeAssistantMessage(role, displayText),
        raw_text: item.raw_text || displayText,
        display_text: displayText,
        analysis_text: item.analysis_text || displayText,
        timestamp: item.timestamp || "",
        persona_name: item.persona_name || "",
        user_persona_name: item.user_persona_name || "",
        speaker_name: item.speaker_name || "",
        vector: item.vector || {},
        predicted_vector: item.predicted_vector || {},
        context_matrix: item.context_matrix || [],
        context_window: item.context_window || [],
        history_flow: item.history_flow || [],
        transition_interpretation: item.transition_interpretation || {},
        has_correction: Boolean(item.has_correction),
        annotation_notes: item.annotation_notes || "",
        annotation_id: item.annotation_id || "",
        pending: true,
      };
    }).filter((item) => item.role === "user" || item.message.trim())
    : [];
  return [...normalizedRows, ...normalizedOptimisticRows];
}

function normalizeAssistantMessage(role, message) {
  const raw = String(message || "");
  if (role === "user") {
    return raw;
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
            const cleaned = stripVisibleReplyLeak(value);
            return cleaned || (looksLikePromptLeak(value) ? "" : value);
          }
        }
      }
    }
  } catch (_error) {
    const cleaned = stripVisibleReplyLeak(raw);
    return cleaned || (looksLikePromptLeak(raw) ? "" : raw);
  }
  const cleaned = stripVisibleReplyLeak(raw);
  return cleaned || (looksLikePromptLeak(raw) ? "" : raw);
}

function looksLikePromptLeak(value) {
  const lowered = String(value || "").trim().toLowerCase();
  if (!lowered) {
    return false;
  }
  return PROMPT_LEAK_MARKERS.some((marker) => lowered.includes(marker));
}

function stripVisibleReplyLeak(value) {
  const raw = String(value || "").trim();
  if (!raw) {
    return "";
  }
  const labeled = extractLabeledReply(raw);
  if (labeled) {
    return labeled;
  }
  const blocks = raw.split(/\n\s*\n/).map((part) => part.trim()).filter(Boolean);
  const cleanBlocks = blocks.filter((block) => !looksLikePromptLeak(block));
  if (cleanBlocks.length && blocks.slice(0, 2).some((block) => looksLikePromptLeak(block))) {
    return cleanBlocks[cleanBlocks.length - 1];
  }
  return looksLikePromptLeak(raw) ? "" : raw;
}

function extractLabeledReply(value) {
  const raw = String(value || "");
  const match = raw.match(/(?:внешний ответ персонажа|ответ персонажа|outer response(?: of the persona)?|corrected line|исправленный вариант реплики)\s*[:：]\s*([\s\S]+)/i);
  let candidate = match?.[1] ? String(match[1] || "").trim() : "";
  let preserveHeading = false;
  if (!candidate) {
    const headed = raw.match(/^\s*(#+\s*(?:answer|ответ)\b[\s\S]+?)(?=\n\s*(?:---|\*\*?\s*review notes?:|\*\*?\s*issues identified:|review notes?:|issues identified:)|$)/i);
    candidate = headed?.[1] ? String(headed[1] || "").trim() : "";
    preserveHeading = Boolean(candidate);
  }
  if (!candidate) {
    return "";
  }
  const stop = candidate.search(/\n\s*(?:[#>*-]+\s*)?(?:внутренняя мысль|inner thought|safety\/policy|constraint|output format|what i can say|what i cannot|что я могу сказать|что я не могу|review notes?|issues identified|role:|task:)\b/i);
  if (stop >= 0) {
    candidate = candidate.slice(0, stop).trim();
  }
  candidate = candidate.replace(/\s*\([^()\n]{1,180}\)\s*$/u, "").trim();
  if (!preserveHeading) {
    candidate = candidate.replace(/^[\-\*\d\.\)\s]+/u, "").trim();
  }
  return looksLikePromptLeak(candidate) ? "" : candidate;
}

function summarizeVector(vector) {
  if (!vector || typeof vector !== "object") return "";
  const labels = [];
  for (const [interpreterId, choice] of Object.entries(vector)) {
    const main = String(choice?.main || "").trim();
    if (!main || HIDDEN_DEFAULTS.has(main)) continue;
    labels.push(`${interpreterId}:${main}`);
    if (labels.length >= 4) break;
  }
  return labels.join(" · ");
}

function formatPrediction(choice) {
  const main = String(choice?.main || "").trim();
  const extra = Array.isArray(choice?.extra) ? choice.extra.filter(Boolean) : [];
  if (!main) return "—";
  return extra.length ? `${main} + ${extra.join(", ")}` : main;
}

function cloneVector(vector) {
  return JSON.parse(JSON.stringify(vector || {}));
}

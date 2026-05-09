import React, { useEffect, useRef, useState } from 'react';

const DEFAULT_DESC = '';
const STORAGE_KEY = 'direct_llm_persona_desc';
const STORAGE_NAME_KEY = 'direct_llm_persona_name';

function Message({ role, content, error }) {
  return (
    <div className={`chat-message ${role === 'user' ? 'chat-message--user' : 'chat-message--assistant'}${error ? ' chat-message--error' : ''}`}>
      <span className="chat-message__role">{role === 'user' ? 'You' : 'LLM'}</span>
      <p className="chat-message__body">{content}</p>
    </div>
  );
}

export function DirectLLMSurface({ personas = [], language = 'en', onRespondDirectLLM, t }) {
  const [personaName, setPersonaName] = useState(
    () => window.localStorage.getItem(STORAGE_NAME_KEY) || ''
  );
  const [desc, setDesc] = useState(
    () => window.localStorage.getItem(STORAGE_KEY) || DEFAULT_DESC
  );
  const [history, setHistory] = useState([]);
  const [input, setInput] = useState('');
  const [running, setRunning] = useState(false);
  const [error, setError] = useState('');
  const bottomRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    window.localStorage.setItem(STORAGE_KEY, desc);
  }, [desc]);

  useEffect(() => {
    window.localStorage.setItem(STORAGE_NAME_KEY, personaName);
  }, [personaName]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [history, running]);

  function applyPreset(name) {
    const found = personas.find((p) => p.name === name);
    if (!found) {
      setPersonaName(name);
      setDesc('');
      return;
    }
    setPersonaName(found.label || found.name || name);
    const knowledge = found.profile?.knowledge || found.persona_object?.knowledge || '';
    const traits = (found.profile?.traits || found.persona_object?.core?.self_image || [])
      .slice(0, 6).join(', ');
    setDesc([traits, knowledge].filter(Boolean).join('. '));
  }

  async function handleSend() {
    const msg = input.trim();
    if (!msg || running) return;
    setError('');
    setInput('');
    const userTurn = { role: 'user', content: msg };
    setHistory((prev) => [...prev, userTurn]);
    setRunning(true);
    try {
      const result = await onRespondDirectLLM({
        message: msg,
        persona_name: personaName.trim() || 'Assistant',
        persona_description: desc.trim(),
        language,
        history: history.slice(-6),
      });
      setHistory((prev) => [...prev, { role: 'assistant', content: result.reply || '...' }]);
    } catch (err) {
      setError(err.message || String(err));
      setHistory((prev) => [...prev, { role: 'assistant', content: `[error] ${err.message || err}`, error: true }]);
    } finally {
      setRunning(false);
      inputRef.current?.focus();
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      void handleSend();
    }
  }

  function handleClear() {
    setHistory([]);
    setError('');
  }

  return (
    <div className="direct-llm-surface">
      {/* Config panel */}
      <div className="direct-llm-config glass-panel">
        <div className="direct-llm-config__row">
          <label className="field-stack compact">
            <span>Persona preset</span>
            <select
              value=""
              onChange={(e) => applyPreset(e.target.value)}
            >
              <option value="">— load from list —</option>
              {personas.map((p) => (
                <option key={p.name} value={p.name}>{p.label || p.name}</option>
              ))}
            </select>
          </label>
          <label className="field-stack compact" style={{ flex: 1 }}>
            <span>Name</span>
            <input
              value={personaName}
              onChange={(e) => setPersonaName(e.target.value)}
              placeholder="e.g. Snape"
            />
          </label>
          <button
            type="button"
            className="button-secondary"
            onClick={handleClear}
            title="Clear conversation"
          >
            Clear
          </button>
        </div>
        <label className="field-stack compact">
          <span>Persona description (sent directly to LLM, no pipeline)</span>
          <textarea
            className="direct-llm-desc"
            rows={3}
            value={desc}
            onChange={(e) => setDesc(e.target.value)}
            placeholder="You are Snape, cold and sarcastic Potions Master. You see through flattery instantly..."
          />
        </label>
      </div>

      {/* Chat thread */}
      <div className="direct-llm-thread">
        {history.length === 0 && (
          <div className="empty-state">
            <p>Enter a persona description above, then type a message.</p>
            <p className="muted">No pipeline, no P-signals — raw LLM only.</p>
          </div>
        )}
        {history.map((turn, i) => (
          <Message key={i} role={turn.role} content={turn.content} error={turn.error} />
        ))}
        {running && (
          <div className="chat-message chat-message--assistant chat-message--pending">
            <span className="chat-message__role">LLM</span>
            <p className="chat-message__body typing-dots">···</p>
          </div>
        )}
        {error && !running && (
          <div className="error-banner" style={{ margin: '0.5rem 0' }}>{error}</div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="direct-llm-composer">
        <textarea
          ref={inputRef}
          className="chat-composer__textarea"
          rows={2}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Message…  (Enter to send, Shift+Enter for newline)"
          disabled={running}
        />
        <button
          type="button"
          className="chat-run-button"
          onClick={() => void handleSend()}
          disabled={running || !input.trim()}
        >
          {running ? '…' : '▶'}
        </button>
      </div>
    </div>
  );
}

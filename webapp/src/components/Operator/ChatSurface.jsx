import React from 'react';

import { ChatGraphPanel } from '../Chat/ChatGraphPanel';

const VECTOR_HIDDEN_DEFAULTS = new Set([
  'absent',
  'neutral',
  'unclear',
  'continuation',
  'defined',
  'direct',
  'literal',
  'statement',
  'non_answer',
  'independent',
  'opening_new_direction',
  'calm',
]);

function renderKeyValues(value) {
  if (!value || typeof value !== 'object') return [];
  return Object.entries(value);
}

function summarizeRuntimeVector(vector) {
  if (!vector || typeof vector !== 'object') return [];
  const labels = [];
  for (const [interpreterId, choice] of Object.entries(vector)) {
    const main = String(choice?.main || '').trim();
    if (!main || VECTOR_HIDDEN_DEFAULTS.has(main)) continue;
    const extras = Array.isArray(choice?.extra) ? choice.extra.filter((item) => item && item !== main).slice(0, 2) : [];
    labels.push(`${interpreterId}=${main}${extras.length ? ` (+${extras.join(', ')})` : ''}`);
    if (labels.length >= 8) break;
  }
  return labels;
}

function SafetyBadge({ result }) {
  if (!result) return null;
  const colors = {
    safe:       'var(--ok)',
    suggestive: 'var(--warn)',
    explicit:   'var(--danger)',
    illegal:    '#ff2244',
  };
  const color = colors[result.label] || 'var(--muted)';
  return (
    <span
      className="safety-badge"
      style={{ '--badge-color': color }}
      title={`Safety: ${result.label} (${(result.confidence * 100).toFixed(0)}% conf) — ${result.action}`}
    >
      <span className="safety-badge-dot" />
      {result.label}
      {result.fast_path ? ' ⚡' : ''}
    </span>
  );
}

function TraceStageTable({ trace }) {
  if (!trace?.stages?.length) {
    return <p className="empty-inline">No trace stages loaded.</p>;
  }
  return (
    <div className="operator-table-wrap">
      <table className="operator-table">
        <thead>
          <tr>
            <th>Stage</th>
            <th>ms</th>
            <th>Meta</th>
          </tr>
        </thead>
        <tbody>
          {trace.stages.map((stage) => (
            <tr key={stage.name}>
              <td>{stage.name}</td>
              <td>{Number(stage.duration_ms ?? 0).toFixed(2)}</td>
              <td>{renderKeyValues(stage.meta || {}).map(([key, entry]) => `${key}=${typeof entry === 'object' ? JSON.stringify(entry) : entry}`).join(' | ') || '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ContextPreview({ preview, t }) {
  if (!preview) {
    return <p className="empty-inline">No context preview yet.</p>;
  }
  const selectedItems = Array.isArray(preview.selected_items) ? preview.selected_items : [];
  const sourceCounts = preview.source_counts || {};
  const vectorRuntime = preview.message_vector_runtime || {};
  const vectorLabels = summarizeRuntimeVector(vectorRuntime.current_vector);
  const flow = Array.isArray(vectorRuntime.history_flow) ? vectorRuntime.history_flow.filter(Boolean) : [];
  const transition = vectorRuntime.transition_interpretation || {};
  return (
    <div className="operator-stack">
      <div className="operator-kv-grid">
        <div><span>Persona</span><strong>{preview.persona_name || '—'}</strong></div>
        <div><span>Entity</span><strong>{preview.current_entity || '—'}</strong></div>
        <div><span>Tokens</span><strong>{preview.estimated_tokens || 0}</strong></div>
        <div><span>Sources</span><strong>{Object.entries(sourceCounts).map(([key, value]) => `${key}:${value}`).join(' | ') || '—'}</strong></div>
      </div>
      {preview.graph_context ? (
        <div className="operator-card-block">
          <strong>Graph context</strong>
          <pre className="operator-pre">{String(preview.graph_context).trim()}</pre>
        </div>
      ) : null}
      {(vectorLabels.length || flow.length || transition.type) ? (
        <div className="operator-card-block">
          <strong>P1..P49 runtime</strong>
          {vectorLabels.length ? <p>{vectorLabels.join(' | ')}</p> : null}
          {flow.length ? <p><strong>{t('annotation_history_flow')}</strong> {flow.join(' → ')}</p> : null}
          {transition.type ? (
            <p>
              <strong>{t('annotation_transition')}</strong>
              {' '}
              {transition.type}
              {Array.isArray(transition.from) && transition.from.length ? ` | from: ${transition.from.join(', ')}` : ''}
              {Array.isArray(transition.to) && transition.to.length ? ` | to: ${transition.to.join(', ')}` : ''}
            </p>
          ) : null}
        </div>
      ) : null}
      <div className="operator-card-block">
        <strong>Selected context items</strong>
        {selectedItems.length ? (
          <ul className="dense-list">
            {selectedItems.slice(0, 10).map((item) => (
              <li key={item.candidate_id || `${item.source}-${item.title}`}>
                <strong>{item.source}</strong>
                {' '}
                {item.title || item.item_type}
                {' '}
                {item.score?.total != null ? `score=${Number(item.score.total).toFixed(3)}` : ''}
                {Array.isArray(item.reasons) && item.reasons.length ? ` | ${item.reasons.join(' | ')}` : ''}
              </li>
            ))}
          </ul>
        ) : <p className="empty-inline">No selected items recorded.</p>}
      </div>
    </div>
  );
}

export function ChatSurface({
  session,
  annotationWorkspace,
  optimisticMessages,
  value,
  onChange,
  running,
  progress,
  composerResetToken,
  onRun,
  lastChatResult,
  activeTrace,
  safetyResult,
  trainingMode,
  onSaveAnnotation,
  onSaveCorrection,
  selectedPersonality,
  personas,
  userPersonaName,
  onUserPersonaChange,
  t,
}) {
  const selection = lastChatResult?.persona_selection || {};
  const response = lastChatResult?.persona_response || {};
  const analysis = lastChatResult?.analysis || {};
  const activePersonaName = [
    selectedPersonality,
    selection.persona_name,
    response.persona_name,
    lastChatResult?.persona_name,
  ].map((value) => String(value || '').trim()).find(Boolean) || '';
  return (
    <div className="operator-surface-grid chat-surface-grid">
      <ChatGraphPanel
        session={session}
        annotationWorkspace={annotationWorkspace}
        optimisticMessages={optimisticMessages}
        value={value}
        onChange={onChange}
        running={running}
        progress={progress}
        composerResetToken={composerResetToken}
        onRun={onRun}
        safetyResult={safetyResult}
        trainingMode={trainingMode}
        onSaveAnnotation={onSaveAnnotation}
        onSaveCorrection={onSaveCorrection}
        activePersonaName={activePersonaName}
        personas={personas}
        userPersonaName={userPersonaName}
        onUserPersonaChange={onUserPersonaChange}
        t={t}
      />
      <aside className="workspace-panel glass-panel operator-inspector-panel">
        <header className="panel-heading compact">
          <div>
            <p className="eyebrow">{t('chat_operator_title')}</p>
            <h2>{t('chat_operator_subtitle')}</h2>
          </div>
        </header>
        <div className="operator-stack">

          {safetyResult ? (
            <section className="operator-card-block safety-panel">
              <h3>{t('chat_safety_label')}</h3>
              <div className="safety-detail-row">
                <SafetyBadge result={safetyResult} />
                <span className="safety-action-tag">{safetyResult.action?.replace(/_/g, ' ')}</span>
                <span className="safety-conf">{(safetyResult.confidence * 100).toFixed(0)}%</span>
              </div>
              {safetyResult.matched_features?.length ? (
                <p className="safety-features">{safetyResult.matched_features.join(' · ')}</p>
              ) : null}
            </section>
          ) : null}

          <section className="operator-card-block">
            <h3>{t('chat_persona_selection')}</h3>
            {lastChatResult ? (
              <ul className="dense-list">
                <li><strong>persona</strong> {selection.persona_name || activePersonaName || '—'}</li>
                <li><strong>source</strong> {selection.source || '—'}</li>
                <li><strong>reason</strong> {selection.reason || '—'}</li>
                {Array.isArray(selection.evidence) && selection.evidence.length ? (
                  <li><strong>evidence</strong> {selection.evidence.join(' | ')}</li>
                ) : null}
              </ul>
            ) : <p className="empty-inline">No persona selection yet.</p>}
          </section>

          <section className="operator-card-block">
            <h3>{t('chat_response_shaping')}</h3>
            {lastChatResult ? (
              <ul className="dense-list">
                <li><strong>style</strong> {response.response_style || '—'}</li>
                <li><strong>situation</strong> {response.situation_summary || '—'}</li>
                <li><strong>reason</strong> {response.reason || '—'}</li>
                {Array.isArray(response.state_influences) && response.state_influences.length ? (
                  <li><strong>state</strong> {response.state_influences.join(' | ')}</li>
                ) : null}
                {Array.isArray(response.trait_influences) && response.trait_influences.length ? (
                  <li><strong>traits</strong> {response.trait_influences.join(' | ')}</li>
                ) : null}
                {Array.isArray(response.learned_influences) && response.learned_influences.length ? (
                  <li><strong>learned</strong> {response.learned_influences.join(' | ')}</li>
                ) : null}
              </ul>
            ) : <p className="empty-inline">No response shaping data yet.</p>}
          </section>

          <section className="operator-card-block">
            <h3>{t('chat_context_preview')}</h3>
            <ContextPreview preview={lastChatResult?.context_preview} t={t} />
          </section>

          <section className="operator-card-block">
            <h3>{t('chat_analysis_preview')}</h3>
            {lastChatResult ? (
              <div className="operator-kv-grid">
                <div><span>tone</span><strong>{analysis?.user_state?.tone || '—'}</strong></div>
                <div><span>intent</span><strong>{analysis?.user_state?.intent || '—'}</strong></div>
                <div><span>language</span><strong>{analysis?.user_state?.language || '—'}</strong></div>
                <div><span>situation</span><strong>{analysis?.situation?.type || '—'}</strong></div>
                <div><span>target</span><strong>{analysis?.situation?.target || '—'}</strong></div>
                <div><span>severity</span><strong>{analysis?.situation?.severity ?? 0}</strong></div>
              </div>
            ) : <p className="empty-inline">No analysis yet.</p>}
          </section>

          <section className="operator-card-block">
            <h3>{t('chat_trace_inspection')}</h3>
            {activeTrace ? (
              <>
                <div className="operator-kv-grid">
                  <div><span>trace</span><strong>{activeTrace.request_id}</strong></div>
                  <div><span>total ms</span><strong>{Number(activeTrace.total_ms ?? 0).toFixed(2)}</strong></div>
                  <div><span>fallback</span><strong>{activeTrace.fallback_used ? activeTrace.fallback_reason || 'yes' : 'no'}</strong></div>
                  <div><span>context tokens</span><strong>{activeTrace.context_tokens || 0}</strong></div>
                </div>
                <TraceStageTable trace={activeTrace} />
              </>
            ) : <p className="empty-inline">No trace loaded yet.</p>}
          </section>
        </div>
      </aside>
    </div>
  );
}

import React, { useState } from 'react';

import { PersonaQuestionnaire } from './PersonaQuestionnaire';

function sectionList(items) {
  return Array.isArray(items) && items.length ? items : null;
}

export function PersonaInspectionSurface({
  personalities,
  selectedPersonality,
  onSelectPersonality,
  personaDetail,
  loading,
  deletingPersonalityId,
  onDeletePersonality,
  onCreatePersonaFromQuestionnaire,
  trainingExamples,
  trainingExamplesLoading,
  onAddTrainingExample,
  onDeleteTrainingExample,
  onExportTrainingExamples,
  lastChatResult,
  t,
}) {
  const revisionHistory = Array.isArray(personaDetail?.revisions?.history) ? personaDetail.revisions.history : [];
  const latestResponse = lastChatResult?.persona_name && selectedPersonality
    && String(lastChatResult.persona_name).toLowerCase() === String(selectedPersonality).toLowerCase()
      ? lastChatResult.persona_response
      : null;

  // Questionnaire state
  const [showQuestionnaire, setShowQuestionnaire] = useState(false);
  const [questionnaireSubmitting, setQuestionnaireSubmitting] = useState(false);
  const [questionnaireError, setQuestionnaireError] = useState('');

  // Training example form state
  const [newInput, setNewInput] = useState('');
  const [newOutput, setNewOutput] = useState('');
  const [newNotes, setNewNotes] = useState('');
  const [addingExample, setAddingExample] = useState(false);
  const [exportContent, setExportContent] = useState('');

  async function handleQuestionnaireSubmit(formData) {
    setQuestionnaireSubmitting(true);
    setQuestionnaireError('');
    try {
      const result = await onCreatePersonaFromQuestionnaire(formData);
      setShowQuestionnaire(false);
      if (result?.persona_slug || result?.persona_name) {
        onSelectPersonality(result.persona_slug || result.persona_name);
      }
    } catch (err) {
      setQuestionnaireError(err.message || String(err));
    } finally {
      setQuestionnaireSubmitting(false);
    }
  }

  const personalityId = personaDetail?.personality?.personality_id
    || personaDetail?.personality_id
    || '';

  async function handleAddExample(e) {
    e.preventDefault();
    if (!newInput.trim() || !newOutput.trim()) return;
    setAddingExample(true);
    try {
      await onAddTrainingExample({
        input_text: newInput.trim(),
        correct_output: newOutput.trim(),
        persona_name: selectedPersonality || '',
        notes: newNotes.trim(),
      });
      setNewInput('');
      setNewOutput('');
      setNewNotes('');
    } finally {
      setAddingExample(false);
    }
  }

  async function handleExport(fmt) {
    const result = await onExportTrainingExamples({ personaName: selectedPersonality || '', fmt });
    if (result?.jsonl) {
      setExportContent(result.jsonl);
    }
  }

  if (showQuestionnaire) {
    return (
      <div className="operator-surface-grid persona-surface-grid">
        <section className="workspace-panel glass-panel qform-panel">
          <PersonaQuestionnaire
            onSubmit={(data) => void handleQuestionnaireSubmit(data)}
            onCancel={() => { setShowQuestionnaire(false); setQuestionnaireError(''); }}
            submitting={questionnaireSubmitting}
            error={questionnaireError}
          />
        </section>
        <section className="workspace-panel glass-panel operator-inspector-panel">
          <header className="panel-heading compact">
            <div>
              <p className="eyebrow">{t('persona_inspection_detail')}</p>
              <h2>Создание новой личности</h2>
            </div>
          </header>
          <div className="operator-stack" style={{ padding: '1.2rem 1rem' }}>
            <p style={{ fontSize: '0.9rem', lineHeight: 1.6 }}>
              Заполни анкету слева. Обязательно только поле <strong>«Псевдоним»</strong>.
            </p>
            <p style={{ fontSize: '0.85rem', opacity: 0.7, lineHeight: 1.6 }}>
              ЛЛМ проанализирует ответы и выведет психологический профиль: цели, страхи, ценности, стиль поведения и речи. Чем больше заполнено — тем точнее личность.
            </p>
            <ul style={{ fontSize: '0.82rem', opacity: 0.65, paddingLeft: '1.2rem', lineHeight: 1.8 }}>
              <li>Идентификация — основные факты</li>
              <li>Характер — как видит себя</li>
              <li>Ценности, цели, страхи — внутренний мир</li>
              <li>Жизненный путь — события и история</li>
              <li>Отношения, общение, привычки</li>
            </ul>
          </div>
        </section>
      </div>
    );
  }

  return (
    <div className="operator-surface-grid persona-surface-grid">
      <section className="workspace-panel glass-panel">
        <header className="panel-heading compact">
          <div>
            <p className="eyebrow">{t('persona_inspection_title')}</p>
            <h2>{t('persona_inspection_list')}</h2>
          </div>
          <button
            type="button"
            className="button-primary qform-create-btn"
            onClick={() => { setShowQuestionnaire(true); setQuestionnaireError(''); }}
          >
            + Анкета
          </button>
        </header>
        <div className="operator-list-stack">
          {personalities.length ? personalities.map((item) => {
            const active = String(selectedPersonality || '').toLowerCase() === String(item.name || '').toLowerCase();
            const pid = item.personality_id || item.name || '';
            const isDeleting = deletingPersonalityId === pid || deletingPersonalityId === item.name;
            return (
              <div
                key={item.name}
                className={`session-card operator-list-card ${active ? 'active' : ''}`}
                style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem' }}
              >
                <button
                  type="button"
                  style={{ flex: 1, background: 'none', border: 'none', padding: 0, textAlign: 'left', cursor: 'pointer', color: 'inherit' }}
                  onClick={() => onSelectPersonality(item.name)}
                  title={item.hover_text || item.short_description || ''}
                >
                  <strong>{item.label || item.name}</strong>
                  <span>{item.readiness || item.entity_type || 'UNKNOWN'} · {item.validation_status || 'partial'}</span>
                  <p>{item.short_description || item.hover_text || Object.entries(item.emotion_vector || {}).map(([key, value]) => `${key}:${value}`).join(' | ')}</p>
                </button>
                {pid ? (
                  <button
                    type="button"
                    className="btn-ghost-danger"
                    disabled={isDeleting}
                    title="Delete personality"
                    onClick={() => {
                      if (window.confirm(`Delete personality "${item.label || item.name}"? This cannot be undone.`)) {
                        onDeletePersonality(pid);
                      }
                    }}
                  >
                    {isDeleting ? '…' : '✕'}
                  </button>
                ) : null}
              </div>
            );
          }) : <div className="empty-inline">No personas available.</div>}
        </div>
      </section>

      <section className="workspace-panel glass-panel operator-inspector-panel">
        <header className="panel-heading compact">
          <div>
            <p className="eyebrow">{t('persona_inspection_detail')}</p>
            <h2>{selectedPersonality || t('chat_personality_none')}</h2>
          </div>
        </header>
        {loading ? (
          <p>{t('persona_loading')}</p>
        ) : personaDetail ? (
          <div className="operator-stack">
            <section className="operator-card-block">
              <h3>{t('persona_baseline')}</h3>
              <div className="operator-kv-grid">
                <div><span>type</span><strong>{personaDetail.baseline?.entity_type || personaDetail.entity_type}</strong></div>
                <div><span>revision</span><strong>{personaDetail.revisions?.baseline_revision || personaDetail.baseline?.revision || 0}</strong></div>
                <div><span>readiness</span><strong>{personaDetail.readiness || personaDetail.structured_persona?.identity?.readiness || 'draft'}</strong></div>
                <div><span>status</span><strong>{personaDetail.structured_persona?.meta?.validation_status || 'partial'}</strong></div>
              </div>
              {personaDetail.short_description || personaDetail.structured_persona?.identity?.short_description ? <p>{personaDetail.short_description || personaDetail.structured_persona?.identity?.short_description}</p> : null}
              {personaDetail.structured_persona?.meta?.hover_text ? <p><strong>hover</strong> {personaDetail.structured_persona.meta.hover_text}</p> : null}
              {sectionList(personaDetail.baseline?.traits || personaDetail.traits) ? <p><strong>traits</strong> {(personaDetail.baseline?.traits || personaDetail.traits).join(' | ')}</p> : null}
              {sectionList(personaDetail.baseline?.aliases) ? <p><strong>aliases</strong> {personaDetail.baseline.aliases.join(' | ')}</p> : null}
              {personaDetail.baseline?.knowledge || personaDetail.knowledge ? <p>{personaDetail.baseline?.knowledge || personaDetail.knowledge}</p> : null}
            </section>

            <section className="operator-card-block">
              <h3>{t('persona_dynamic_state')}</h3>
              <div className="operator-kv-grid">
                <div><span>maturity</span><strong>{personaDetail.indicators?.maturity_level || 'bootstrap'}</strong></div>
                <div><span>confidence</span><strong>{personaDetail.indicators?.confidence_score ?? 0}</strong></div>
                <div><span>evidence</span><strong>{personaDetail.indicators?.evidence_count ?? 0}</strong></div>
                <div><span>locked</span><strong>{String(personaDetail.indicators?.adaptation_locked ?? false)}</strong></div>
              </div>
              <p><strong>emotion</strong> {Object.entries(personaDetail.dynamic_state?.emotion_vector || personaDetail.emotion_vector || {}).map(([key, value]) => `${key}:${value}`).join(' | ')}</p>
              {personaDetail.dynamic_state?.last_situation ? <p><strong>last situation</strong> {personaDetail.dynamic_state.last_situation}</p> : null}
              {personaDetail.dynamic_state?.last_response_style ? <p><strong>last style</strong> {personaDetail.dynamic_state.last_response_style}</p> : null}
            </section>

            <section className="operator-card-block">
              <h3>{t('persona_learned_patterns')}</h3>
              {sectionList(personaDetail.learned_patterns?.learned_traits) ? <p><strong>learned traits</strong> {personaDetail.learned_patterns.learned_traits.join(' | ')}</p> : null}
              {sectionList(personaDetail.learned_patterns?.examples || personaDetail.examples) ? (
                <p><strong>examples</strong> {(personaDetail.learned_patterns?.examples || personaDetail.examples).slice(0, 6).join(' | ')}</p>
              ) : null}
              {personaDetail.structured_persona ? (
                <p><strong>structured</strong> {Object.entries(personaDetail.structured_persona).map(([key, value]) => `${key}=${Array.isArray(value) ? value.join(', ') : typeof value === 'object' ? Object.keys(value || {}).join(', ') : value}`).join(' | ')}</p>
              ) : null}
              {personaDetail.learned_patterns?.persona_form ? (
                <p><strong>form</strong> {Object.entries(personaDetail.learned_patterns.persona_form).map(([key, value]) => `${key}=${Array.isArray(value) ? value.join(', ') : value}`).join(' | ')}</p>
              ) : null}
              {personaDetail.learned_patterns?.decision_explanation || personaDetail.triad?.decision_explanation ? (
                <p><strong>decision</strong> {personaDetail.learned_patterns?.decision_explanation || personaDetail.triad?.decision_explanation}</p>
              ) : null}
            </section>

            <section className="operator-card-block">
              <h3>{t('persona_revision_history')}</h3>
              {revisionHistory.length ? (
                <ul className="dense-list">
                  {revisionHistory.slice(0, 8).map((item) => (
                    <li key={`${item.revision}-${item.timestamp}`}>
                      <strong>r{item.revision}</strong>
                      {' '}
                      {item.reason || 'update'}
                      {' '}
                      <span>{item.timestamp || ''}</span>
                    </li>
                  ))}
                </ul>
              ) : <p className="empty-inline">No revision history.</p>}
            </section>

            {latestResponse ? (
              <section className="operator-card-block">
                <h3>{t('persona_latest_runtime')}</h3>
                <p><strong>style</strong> {latestResponse.response_style || '—'}</p>
                <p>{latestResponse.reason || '—'}</p>
              </section>
            ) : null}

            {/* Training examples panel */}
            <section className="operator-card-block">
              <h3>Training examples</h3>
              <p className="eyebrow" style={{ marginBottom: '0.5rem' }}>
                Save correct responses to curate a fine-tuning dataset for this persona.
              </p>

              <form onSubmit={(e) => void handleAddExample(e)} style={{ marginBottom: '1rem' }}>
                <div style={{ marginBottom: '0.4rem' }}>
                  <label style={{ display: 'block', fontSize: '0.75rem', marginBottom: '0.2rem' }}>Input (user message)</label>
                  <textarea
                    value={newInput}
                    onChange={(e) => setNewInput(e.target.value)}
                    placeholder="What the user said…"
                    rows={2}
                    style={{ width: '100%', resize: 'vertical', fontSize: '0.8rem' }}
                    required
                  />
                </div>
                <div style={{ marginBottom: '0.4rem' }}>
                  <label style={{ display: 'block', fontSize: '0.75rem', marginBottom: '0.2rem' }}>Correct output (ideal response)</label>
                  <textarea
                    value={newOutput}
                    onChange={(e) => setNewOutput(e.target.value)}
                    placeholder="How the persona should have responded…"
                    rows={3}
                    style={{ width: '100%', resize: 'vertical', fontSize: '0.8rem' }}
                    required
                  />
                </div>
                <div style={{ marginBottom: '0.6rem' }}>
                  <label style={{ display: 'block', fontSize: '0.75rem', marginBottom: '0.2rem' }}>Notes (optional)</label>
                  <input
                    type="text"
                    value={newNotes}
                    onChange={(e) => setNewNotes(e.target.value)}
                    placeholder="Why this is a good example…"
                    style={{ width: '100%', fontSize: '0.8rem' }}
                  />
                </div>
                <button type="submit" className="btn-primary" disabled={addingExample || !newInput.trim() || !newOutput.trim()}>
                  {addingExample ? 'Saving…' : 'Save example'}
                </button>
              </form>

              {trainingExamplesLoading ? (
                <p className="empty-inline">Loading…</p>
              ) : (Array.isArray(trainingExamples) && trainingExamples.length > 0) ? (
                <>
                  <ul className="dense-list" style={{ marginBottom: '0.75rem' }}>
                    {trainingExamples.map((ex) => (
                      <li key={ex.example_id} style={{ display: 'flex', gap: '0.5rem', alignItems: 'flex-start', padding: '0.4rem 0', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                        <div style={{ flex: 1, fontSize: '0.8rem' }}>
                          <div><strong>In:</strong> {ex.input_text?.slice(0, 120)}{ex.input_text?.length > 120 ? '…' : ''}</div>
                          <div style={{ color: 'var(--text-secondary, #aaa)' }}><strong>Out:</strong> {ex.correct_output?.slice(0, 120)}{ex.correct_output?.length > 120 ? '…' : ''}</div>
                          {ex.notes ? <div style={{ fontSize: '0.72rem', opacity: 0.6 }}>{ex.notes}</div> : null}
                          <div style={{ fontSize: '0.72rem', opacity: 0.5 }}>{ex.created_at}</div>
                        </div>
                        <button
                          type="button"
                          className="btn-ghost-danger"
                          onClick={() => onDeleteTrainingExample(ex.example_id)}
                          title="Delete this example"
                        >
                          ✕
                        </button>
                      </li>
                    ))}
                  </ul>
                  <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                    <button type="button" className="btn-secondary" onClick={() => void handleExport('openai')}>
                      Export OpenAI JSONL
                    </button>
                    <button type="button" className="btn-secondary" onClick={() => void handleExport('simple')}>
                      Export simple JSONL
                    </button>
                  </div>
                  {exportContent ? (
                    <textarea
                      readOnly
                      value={exportContent}
                      rows={8}
                      style={{ width: '100%', marginTop: '0.5rem', fontSize: '0.72rem', resize: 'vertical', fontFamily: 'monospace' }}
                      onClick={(e) => e.target.select()}
                    />
                  ) : null}
                </>
              ) : (
                <p className="empty-inline">No training examples yet. Add the first one above.</p>
              )}
            </section>
          </div>
        ) : (
          <p>{t('persona_select_prompt')}</p>
        )}
      </section>
    </div>
  );
}

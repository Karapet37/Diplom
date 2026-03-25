import React from 'react';

function sectionList(items) {
  return Array.isArray(items) && items.length ? items : null;
}

export function PersonaInspectionSurface({
  personalities,
  selectedPersonality,
  onSelectPersonality,
  personaDetail,
  loading,
  lastChatResult,
  t,
}) {
  const revisionHistory = Array.isArray(personaDetail?.revisions?.history) ? personaDetail.revisions.history : [];
  const latestResponse = lastChatResult?.persona_name && selectedPersonality
    && String(lastChatResult.persona_name).toLowerCase() === String(selectedPersonality).toLowerCase()
      ? lastChatResult.persona_response
      : null;

  return (
    <div className="operator-surface-grid persona-surface-grid">
      <section className="workspace-panel glass-panel">
        <header className="panel-heading compact">
          <div>
            <p className="eyebrow">{t('persona_inspection_title')}</p>
            <h2>{t('persona_inspection_list')}</h2>
          </div>
        </header>
        <div className="operator-list-stack">
          {personalities.length ? personalities.map((item) => {
            const active = String(selectedPersonality || '').toLowerCase() === String(item.name || '').toLowerCase();
            return (
              <button
                key={item.name}
                type="button"
                className={`session-card operator-list-card ${active ? 'active' : ''}`}
                onClick={() => onSelectPersonality(item.name)}
              >
                <strong>{item.name}</strong>
                <span>{item.entity_type || 'UNKNOWN'}</span>
                <p>{Object.entries(item.emotion_vector || {}).map(([key, value]) => `${key}:${value}`).join(' | ')}</p>
              </button>
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
              </div>
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
          </div>
        ) : (
          <p>{t('persona_select_prompt')}</p>
        )}
      </section>
    </div>
  );
}

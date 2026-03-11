import React from "react";

function probability(value) {
  return `${Math.round(Number(value || 0) * 100)}%`;
}

export function HypothesisPanel({ value, onChange, running, onRun, interpretResult, selectedIndex, onSelectHypothesis, plannerPreview, t }) {
  const topHypotheses = interpretResult?.top_hypotheses || [];

  return (
    <section className="workspace-panel glass-panel">
      <header className="panel-heading">
        <div>
          <p className="eyebrow">{t("hypothesis_eyebrow")}</p>
          <h2>{t("hypothesis_title")}</h2>
          <p>{t("hypothesis_subtitle")}</p>
        </div>
        <button type="button" onClick={onRun} disabled={running}>
          {running ? t("hypothesis_running") : t("hypothesis_run")}
        </button>
      </header>

      <div className="panel-form grid-two">
        <label className="field-stack">
          <span>{t("hypothesis_message")}</span>
          <textarea value={value.text} onChange={(event) => onChange({ ...value, text: event.target.value })} />
        </label>
        <label className="field-stack compact">
          <span>{t("hypothesis_count")}</span>
          <input type="number" min="2" max="6" value={value.k} onChange={(event) => onChange({ ...value, k: Number(event.target.value || 3) })} />
          <small>{t("hypothesis_help")}</small>
        </label>
      </div>

      {!interpretResult ? (
        <div className="empty-state">
          <h3>{t("hypothesis_empty_title")}</h3>
          <p>{t("hypothesis_empty_text")}</p>
        </div>
      ) : (
        <>
          <div className="summary-strip">
            <article className="summary-mini-card">
              <span>{t("hypothesis_uncertainty")}</span>
              <strong>{probability(interpretResult?.uncertainty || 0)}</strong>
            </article>
            <article className="summary-mini-card">
              <span>{t("hypothesis_selected_branch")}</span>
              <strong>{selectedIndex >= 0 ? `#${selectedIndex + 1}` : t("sidebar_none")}</strong>
            </article>
            <article className="summary-mini-card wide">
              <span>{t("hypothesis_clarifying_question")}</span>
              <strong>{interpretResult?.best_clarifying_question?.question || t("sidebar_none")}</strong>
            </article>
          </div>
          <div className="hypothesis-grid">
            {topHypotheses.map((hypothesis, index) => {
              const selected = selectedIndex === index;
              return (
                <article
                  key={`${hypothesis.id || hypothesis.label}-${index}`}
                  className={`hypothesis-card ${selected ? "selected" : ""}`}
                  onClick={() =>
                    onSelectHypothesis(index, {
                      ...hypothesis,
                      evidence: hypothesis.evidence || [],
                      contradictions: hypothesis.contradictions || [],
                      predicted_actions: plannerPreview?.best_line || [],
                    })
                  }
                >
                  <header>
                    <span className="hypothesis-index">H{index + 1}</span>
                    <span className="hypothesis-score">{probability(hypothesis.probability)}</span>
                  </header>
                  <h3>{hypothesis.label || hypothesis.id || `${t("sidebar_hypotheses")} ${index + 1}`}</h3>
                  <p>{hypothesis.description || hypothesis.claim || t("inspector_no_claim")}</p>
                  <div className="hypothesis-subblock">
                    <strong>{t("inspector_supporting_cues")}</strong>
                    <ul className="dense-list compact">
                      {(hypothesis.evidence || []).slice(0, 4).map((item, cueIndex) => (
                        <li key={`${cueIndex}-${item.cue || item}`}>{item.cue ? `${item.cue}: ${item.evidence}` : String(item)}</li>
                      ))}
                    </ul>
                  </div>
                </article>
              );
            })}
          </div>
        </>
      )}
    </section>
  );
}

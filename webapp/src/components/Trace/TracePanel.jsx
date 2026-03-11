import React from "react";

function TraceNode({ node, depth = 0, onSelect }) {
  if (!node) {
    return null;
  }
  const score = Number(node.expected_score ?? node.score ?? 0).toFixed(3);
  const label = node.best_action || node.type || "trace";
  return (
    <details className="trace-node" open={depth < 2}>
      <summary onClick={(event) => { event.preventDefault(); onSelect(node); }}>
        <div>
          <strong>{label}</strong>
          <span>{node.type || "node"}</span>
        </div>
        <span className="trace-score">{score}</span>
      </summary>
      <div className="trace-node__content">
        {(node.children || []).map((child, index) => (
          <TraceNode key={`child-${depth}-${index}`} node={child} depth={depth + 1} onSelect={onSelect} />
        ))}
        {(node.chance || []).map((chance, index) => (
          <div key={`chance-${depth}-${index}`} className="trace-chance" onClick={() => onSelect({ ...chance.child, assumptions: [chance.hypothesis_id, `p=${chance.probability}`] })}>
            <header>
              <strong>{chance.hypothesis_id}</strong>
              <span>{Math.round(Number(chance.probability || 0) * 100)}%</span>
            </header>
            <p>{JSON.stringify(chance.transition?.deltas || {})}</p>
            <TraceNode node={chance.child} depth={depth + 1} onSelect={onSelect} />
          </div>
        ))}
      </div>
    </details>
  );
}

export function TracePanel({ value, onChange, running, onRun, planResult, onSelectTrace, t }) {
  const bestLine = planResult?.best_line || [];
  const alternatives = planResult?.alternatives || [];
  const assumptions = planResult?.assumptions || {};

  return (
    <section className="workspace-panel glass-panel">
      <header className="panel-heading">
        <div>
          <p className="eyebrow">{t("trace_eyebrow")}</p>
          <h2>{t("trace_title")}</h2>
          <p>{t("trace_subtitle")}</p>
        </div>
        <button type="button" onClick={onRun} disabled={running}>
          {running ? t("trace_running") : t("trace_run")}
        </button>
      </header>

      <div className="panel-form grid-three">
        <label className="field-stack">
          <span>{t("trace_goal")}</span>
          <input value={value.goal} onChange={(event) => onChange({ ...value, goal: event.target.value })} />
        </label>
        <label className="field-stack compact">
          <span>{t("trace_depth")}</span>
          <input type="number" min="1" max="5" value={value.depth} onChange={(event) => onChange({ ...value, depth: Number(event.target.value || 3) })} />
        </label>
        <label className="field-stack compact">
          <span>{t("trace_beam_width")}</span>
          <input type="number" min="1" max="8" value={value.beamWidth} onChange={(event) => onChange({ ...value, beamWidth: Number(event.target.value || 4) })} />
        </label>
        <label className="field-stack span-3">
          <span>{t("trace_source_text")}</span>
          <textarea value={value.text} onChange={(event) => onChange({ ...value, text: event.target.value })} />
        </label>
      </div>

      {!planResult ? (
        <div className="empty-state">
          <h3>{t("trace_empty_title")}</h3>
          <p>{t("trace_empty_text")}</p>
        </div>
      ) : (
        <>
          <div className="summary-strip">
            <article className="summary-mini-card wide">
              <span>{t("trace_best_line")}</span>
              <strong>{bestLine.join(" → ") || t("trace_no_actions")}</strong>
            </article>
            <article className="summary-mini-card">
              <span>{t("inspector_score")}</span>
              <strong>{Number(planResult.best_score || 0).toFixed(3)}</strong>
            </article>
            <article className="summary-mini-card">
              <span>{t("trace_visited")}</span>
              <strong>{planResult.nodes_visited || 0}</strong>
            </article>
            <article className="summary-mini-card wide">
              <span>{t("hypothesis_clarifying_question")}</span>
              <strong>{planResult.interpretation?.best_clarifying_question?.question || t("sidebar_none")}</strong>
            </article>
          </div>

          <div className="trace-layout">
            <section className="trace-column">
              <header><h3>{t("trace_alternatives")}</h3></header>
              {!alternatives.length ? <div className="empty-inline">{t("trace_no_alternatives")}</div> : null}
              {alternatives.map((alternative, index) => (
                <article key={`alt-${index}`} className="alternative-card" onClick={() => onSelectTrace({ type: "alternative", score: alternative.expected_score, assumptions: alternative.line, best_action: alternative.first_action })}>
                  <strong>{alternative.first_action || `${t("trace_alternative")} ${index + 1}`}</strong>
                  <span>{Number(alternative.expected_score || 0).toFixed(3)}</span>
                  <p>{(alternative.line || []).join(" → ")}</p>
                </article>
              ))}
              <header><h3>{t("inspector_assumptions")}</h3></header>
              <div className="assumption-list">
                {Object.entries(assumptions).map(([key, value]) => (
                  <div key={key} className="assumption-row">
                    <span>{key}</span>
                    <strong>{Array.isArray(value) ? value.join(", ") : String(value)}</strong>
                  </div>
                ))}
              </div>
            </section>
            <section className="trace-column wide">
              <header><h3>{t("trace_tree")}</h3></header>
              <div className="trace-tree-shell">
                <TraceNode node={planResult.trace} onSelect={onSelectTrace} />
              </div>
            </section>
          </div>
        </>
      )}
    </section>
  );
}

import React from "react";

export function IngestPanel({
  value,
  onChange,
  runningIngest,
  runningRebuild,
  onIngest,
  onRebuild,
  onImportFile,
  ingestResult,
  rebuildResult,
  loops,
  graphMetrics,
  t,
}) {
  return (
    <section className="workspace-panel glass-panel">
      <header className="panel-heading">
        <div>
          <p className="eyebrow">{t("ingest_eyebrow")}</p>
          <h2>{t("ingest_title")}</h2>
          <p>{t("ingest_subtitle")}</p>
        </div>
        <div className="header-actions">
          <label className="button-secondary pseudo-file">
            {t("ingest_import_source")}
            <input type="file" accept=".txt,.md,.json" onChange={onImportFile} hidden />
          </label>
          <button type="button" className="button-secondary" onClick={onRebuild} disabled={runningRebuild}>
            {runningRebuild ? t("ingest_rebuilding") : t("ingest_rebuild")}
          </button>
          <button type="button" onClick={onIngest} disabled={runningIngest}>
            {runningIngest ? t("ingest_running") : t("ingest_run")}
          </button>
        </div>
      </header>

      <div className="panel-form grid-two">
        <label className="field-stack compact">
          <span>{t("ingest_source_id")}</span>
          <input value={value.sourceId} onChange={(event) => onChange({ ...value, sourceId: event.target.value })} placeholder="src:field-note" />
        </label>
        <div className="status-stack">
          <div className="summary-mini-card"><span>{t("ingest_current_nodes")}</span><strong>{graphMetrics.nodeCount}</strong></div>
          <div className="summary-mini-card"><span>{t("ingest_current_edges")}</span><strong>{graphMetrics.edgeCount}</strong></div>
          <div className="summary-mini-card"><span>{t("ingest_loop_warnings")}</span><strong>{loops.count || 0}</strong></div>
        </div>
        <label className="field-stack span-2">
          <span>{t("ingest_raw_text")}</span>
          <textarea value={value.text} onChange={(event) => onChange({ ...value, text: event.target.value })} />
        </label>
      </div>

      <div className="controller-grid split">
        <section className="controller-card wide">
          <h3>{t("ingest_latest_ingest")}</h3>
          {ingestResult ? (
            <div className="dense-list compact">
              <div className="flag-row"><span>{t("ingest_source_id")}</span><strong>{ingestResult.source_id}</strong></div>
              <div className="flag-row"><span>{t("sidebar_nodes")}</span><strong>{ingestResult.nodes}</strong></div>
              <div className="flag-row"><span>{t("sidebar_edges")}</span><strong>{ingestResult.edges}</strong></div>
              <div className="flag-row"><span>{t("inspector_evidence_count")}</span><strong>{ingestResult.evidence}</strong></div>
            </div>
          ) : (
            <div className="empty-inline">{t("ingest_no_ingest")}</div>
          )}
        </section>
        <section className="controller-card wide">
          <h3>{t("ingest_latest_rebuild")}</h3>
          {rebuildResult ? (
            <div className="dense-list compact">
              <div className="flag-row"><span>{t("inspector_mode")}</span><strong>{rebuildResult.mode || "full"}</strong></div>
              <div className="flag-row"><span>{t("ingest_sources_processed")}</span><strong>{rebuildResult.sources_processed || 0}</strong></div>
              <div className="flag-row"><span>{t("sidebar_nodes")}</span><strong>{rebuildResult.nodes || 0}</strong></div>
              <div className="flag-row"><span>{t("sidebar_edges")}</span><strong>{rebuildResult.edges || 0}</strong></div>
            </div>
          ) : (
            <div className="empty-inline">{t("ingest_no_rebuild")}</div>
          )}
        </section>
      </div>

      <section className="controller-card wide warning-card">
        <h3>{t("ingest_consistency_warnings")}</h3>
        {loops.count ? (
          <ul className="dense-list compact">
            {(loops.self_loops || []).slice(0, 6).map((edge, index) => (
              <li key={`self-${index}`}>Self loop on {edge.src_id}</li>
            ))}
            {(loops.two_cycles || []).slice(0, 6).map((cycle, index) => (
              <li key={`cycle-${index}`}>Two-cycle between {cycle.a} and {cycle.b}</li>
            ))}
          </ul>
        ) : (
          <div className="empty-inline">{t("graph_no_loops")}</div>
        )}
      </section>
    </section>
  );
}

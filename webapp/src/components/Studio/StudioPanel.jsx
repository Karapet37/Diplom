import React from "react";

function humanizeDatasetId(datasetId) {
  return String(datasetId || "").replace(/_/g, " ");
}

function ActivityItem({ item }) {
  return (
    <article className="activity-item">
      <header>
        <strong>{item.title}</strong>
        <span>{item.timestamp}</span>
      </header>
      <p>{item.message}</p>
    </article>
  );
}

export function StudioPanel({
  datasetOptions,
  selectedDatasetId,
  onDatasetChange,
  onLoadFoundation,
  onImportGraphFile,
  seedRunning,
  graphMetrics,
  loops,
  sources,
  activityFeed,
  t,
}) {
  return (
    <section className="workspace-panel glass-panel">
      <header className="panel-heading">
        <div>
          <p className="eyebrow">{t("foundations_eyebrow")}</p>
          <h2>{t("foundations_title")}</h2>
          <p>{t("foundations_text")}</p>
        </div>
      </header>

      <div className="studio-grid">
        <section className="controller-card">
          <h3>{t("foundations_load")}</h3>
          <label className="field-stack compact">
            <span>{t("foundations_dataset")}</span>
            <select value={selectedDatasetId} onChange={(event) => onDatasetChange(event.target.value)}>
              {datasetOptions.map((datasetId) => (
                <option key={datasetId} value={datasetId}>
                  {humanizeDatasetId(datasetId)}
                </option>
              ))}
            </select>
          </label>
          <div className="header-actions">
            <button type="button" onClick={onLoadFoundation} disabled={seedRunning || !selectedDatasetId}>
              {seedRunning ? t("foundations_loading") : t("foundations_load_button")}
            </button>
            <label className="button-secondary pseudo-file">
              {t("foundations_import")}
              <input type="file" accept=".json" hidden onChange={onImportGraphFile} />
            </label>
          </div>
          <p className="panel-copy">{t("foundations_panel_text")}</p>
        </section>

        <section className="controller-card">
          <h3>{t("foundations_snapshot")}</h3>
          <div className="flag-row"><span>{t("sidebar_nodes")}</span><strong>{graphMetrics.nodeCount}</strong></div>
          <div className="flag-row"><span>{t("sidebar_edges")}</span><strong>{graphMetrics.edgeCount}</strong></div>
          <div className="flag-row"><span>{t("sidebar_loops")}</span><strong>{loops.count || 0}</strong></div>
          <div className="flag-row"><span>{t("foundations_sources")}</span><strong>{sources.length}</strong></div>
        </section>
      </div>

      <section className="controller-card wide">
        <h3>{t("foundations_recent_changes")}</h3>
        {activityFeed.length ? (
          <div className="activity-list">
            {activityFeed.map((item) => <ActivityItem key={item.id} item={item} />)}
          </div>
        ) : (
          <div className="empty-inline">{t("foundations_no_changes")}</div>
        )}
      </section>
    </section>
  );
}

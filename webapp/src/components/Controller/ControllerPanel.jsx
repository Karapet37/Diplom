import React from "react";

function FlagRow({ label, value, t }) {
  return (
    <div className="flag-row">
      <span>{label}</span>
      <strong>{value ? t("inspector_enabled") : t("inspector_disabled")}</strong>
    </div>
  );
}

export function ControllerPanel({
  health,
  status,
  controlState,
  modules,
  llmPolicy,
  modelAdvisors,
  qualityReport,
  events,
  operations,
  onRunQuality,
  qualityRunning,
  t,
}) {
  const flags = controlState?.flags || {};
  const advisorPayload = modelAdvisors?.advisors || modelAdvisors || {};
  const detectedModels = advisorPayload.detected_models || advisorPayload.models || [];
  const prompts = modelAdvisors?.prompts || [];

  return (
    <section className="workspace-panel glass-panel">
      <header className="panel-heading">
        <div>
          <p className="eyebrow">{t("controller_eyebrow")}</p>
          <h2>{t("controller_title")}</h2>
          <p>{t("controller_subtitle")}</p>
        </div>
        <button type="button" className="button-secondary" onClick={onRunQuality} disabled={qualityRunning}>
          {qualityRunning ? t("controller_checking") : t("controller_refresh_integrity")}
        </button>
      </header>

      <div className="controller-grid">
        <section className="controller-card">
          <h3>{t("controller_permissions")}</h3>
          <FlagRow label={t("inspector_read_only")} value={flags.read_only} t={t} />
          <FlagRow label={t("inspector_graph_writes")} value={flags.allow_graph_writes} t={t} />
          <FlagRow label={t("controller_project_daily")} value={flags.allow_project_daily} t={t} />
          <FlagRow label={t("inspector_prompt_execution")} value={flags.allow_prompt_execution} t={t} />
          <FlagRow label={t("controller_knowledge_mutations")} value={flags.allow_knowledge_mutations} t={t} />
        </section>

        <section className="controller-card">
          <h3>{t("controller_runtime_health")}</h3>
          <div className="flag-row"><span>{t("controller_service")}</span><strong>{health?.service || "unknown"}</strong></div>
          <div className="flag-row"><span>{t("controller_storage")}</span><strong>{status?.storage_adapter || "unknown"}</strong></div>
          <div className="flag-row"><span>{t("controller_auth")}</span><strong>{health?.security?.auth_enabled ? t("inspector_enabled") : t("inspector_off")}</strong></div>
          <div className="flag-row"><span>{t("controller_rate_limit")}</span><strong>{health?.security?.rate_limit_enabled ? t("inspector_enabled") : t("inspector_off")}</strong></div>
          <div className="flag-row"><span>{t("controller_admin_key")}</span><strong>{controlState?.admin_key_configured ? t("controller_configured") : t("controller_not_set")}</strong></div>
        </section>

        <section className="controller-card">
          <h3>{t("controller_llm")}</h3>
          <div className="flag-row"><span>{t("controller_policy_mode")}</span><strong>{llmPolicy?.policy?.mode || llmPolicy?.mode || "unknown"}</strong></div>
          <div className="flag-row"><span>{t("controller_detected_models")}</span><strong>{detectedModels.length}</strong></div>
          <div className="flag-row"><span>{t("controller_prompt_catalog")}</span><strong>{prompts.length}</strong></div>
          <div className="flag-row"><span>{t("controller_translator_policy")}</span><strong>{advisorPayload.translator_policy || "unknown"}</strong></div>
        </section>

        <section className="controller-card wide">
          <h3>{t("controller_integrity_report")}</h3>
          {qualityReport ? (
            <>
              <div className="flag-row"><span>{t("inspector_score")}</span><strong>{qualityReport.score}</strong></div>
              <div className="flag-row"><span>{t("sidebar_nodes")}</span><strong>{qualityReport.checks?.node_count || 0}</strong></div>
              <div className="flag-row"><span>{t("sidebar_edges")}</span><strong>{qualityReport.checks?.edge_count || 0}</strong></div>
              <div className="flag-row"><span>{t("controller_contradictions")}</span><strong>{qualityReport.checks?.contradictions || 0}</strong></div>
              <div className="flag-row"><span>{t("controller_orphans")}</span><strong>{qualityReport.checks?.orphan_nodes || 0}</strong></div>
              <div className="flag-row"><span>{t("controller_weak_nodes")}</span><strong>{qualityReport.checks?.weak_descriptions || 0}</strong></div>
              <div className="flag-row"><span>{t("controller_missing_fields")}</span><strong>{qualityReport.checks?.missing_practical_fields || 0}</strong></div>
              <div className="flag-row"><span>{t("controller_duplicates")}</span><strong>{qualityReport.checks?.duplicate_candidates || 0}</strong></div>
              <ul className="dense-list compact">
                {(qualityReport.recommendations || []).map((item, index) => (
                  <li key={`${index}-${item}`}>{String(item)}</li>
                ))}
              </ul>
              {(qualityReport.weak_nodes || []).length ? (
                <>
                  <h4>{t("controller_weak_nodes_list")}</h4>
                  <ul className="dense-list compact">
                    {(qualityReport.weak_nodes || []).slice(0, 6).map((item) => (
                      <li key={item.id}>
                        <strong>{item.name}</strong>
                        <span>{item.type}</span>
                      </li>
                    ))}
                  </ul>
                </>
              ) : null}
              {(qualityReport.duplicate_candidates || []).length ? (
                <>
                  <h4>{t("controller_duplicate_groups")}</h4>
                  <ul className="dense-list compact">
                    {(qualityReport.duplicate_candidates || []).slice(0, 6).map((item, index) => (
                      <li key={`${item.type}-${item.name}-${index}`}>
                        <strong>{item.name}</strong>
                        <span>{item.type} · {item.count}</span>
                      </li>
                    ))}
                  </ul>
                </>
              ) : null}
            </>
          ) : (
            <div className="empty-inline">{t("controller_no_integrity")}</div>
          )}
        </section>
      </div>

      <div className="controller-grid split">
        <section className="controller-card wide">
          <h3>{t("controller_recent_operations")}</h3>
          {!operations.length ? <div className="empty-inline">{t("controller_no_operations")}</div> : null}
          <div className="queue-list">
            {operations.map((item) => (
              <article key={item.id} className={`queue-item ${item.status}`}>
                <header>
                  <strong>{item.label}</strong>
                  <span>{item.status}</span>
                </header>
                <p>{item.message}</p>
              </article>
            ))}
          </div>
        </section>
        <section className="controller-card wide">
          <h3>{t("controller_event_stream")}</h3>
          {!events.length ? <div className="empty-inline">{t("controller_no_events")}</div> : null}
          <ul className="dense-list compact mono-list">
            {events.slice(0, 12).map((event) => (
              <li key={event.id}>
                <strong>{event.event_type}</strong>
                <span>{new Date((event.timestamp || 0) * 1000).toLocaleTimeString()}</span>
              </li>
            ))}
          </ul>
        </section>
      </div>

      <div className="controller-grid split">
        <section className="controller-card wide">
          <h3>{t("controller_modules")}</h3>
          {!modules.length ? <div className="empty-inline">{t("controller_no_modules")}</div> : null}
          <div className="pill-grid">
            {modules.map((moduleName) => (
              <span key={moduleName} className="pill muted">{moduleName}</span>
            ))}
          </div>
        </section>
        <section className="controller-card wide">
          <h3>{t("controller_detected_models")}</h3>
          {!detectedModels.length ? <div className="empty-inline">{t("controller_no_models")}</div> : null}
          <ul className="dense-list compact">
            {detectedModels.slice(0, 10).map((item, index) => (
              <li key={index}>{typeof item === "string" ? item : item.path || item.name || JSON.stringify(item)}</li>
            ))}
          </ul>
        </section>
      </div>
    </section>
  );
}

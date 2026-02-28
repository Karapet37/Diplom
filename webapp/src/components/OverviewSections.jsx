import React from "react";

export function DemoOverviewSection({
  t,
  busy,
  interactionModePanel,
  demoNarrative,
  onDemoNarrativeChange,
  onSeedDemo,
  onOpenMultitool,
  demoResult,
  graphNodeCount,
  graphElement,
  onRunPersonalTreeIngest,
  onRefreshDrafts,
  executionPanel,
  reviewPanel,
}) {
  return (
    <section className="card grid-2">
      <div>
        {interactionModePanel}
        <h2>{t("overview_section_demo")}</h2>
        <div className="row">
          <label>{t("demo_narrative")}</label>
          <textarea
            value={demoNarrative}
            onChange={(e) => onDemoNarrativeChange(e.target.value)}
            rows={8}
            placeholder={t("demo_narrative_placeholder")}
          />
        </div>
        <div className="row-actions">
          <button disabled={busy} onClick={onSeedDemo}>
            {t("action_seed_demo")}
          </button>
          <button type="button" disabled={busy || !demoResult} onClick={onOpenMultitool}>
            {t("multitool_title")}
          </button>
        </div>
      </div>
      <div className="multitool-layout">
        <div className="multitool-dashboard">
          <h2>{t("reasoning_chain")}</h2>
          {demoResult ? (
            <>
              <div className="insight-metrics-grid">
                <div className="insight-metric-card">
                  <strong>{String(demoResult?.demo?.persona_name || "You")}</strong>
                  <span>{String(demoResult?.demo?.mode || "fallback")}</span>
                </div>
                <div className="insight-metric-card">
                  <strong>{t("graph_visualization")}</strong>
                  <span>{graphNodeCount}</span>
                </div>
              </div>
              <ul className="insight-list">
                <li>
                  <strong>{String(demoResult?.demo?.persona_name || "You")}</strong>
                  <span>{String(demoResult?.demo?.narrative || "").trim() || "-"}</span>
                </li>
                {String(demoResult?.demo?.llm_error || "").trim() ? (
                  <li>
                    <strong>{t("log_error")}</strong>
                    <span>{String(demoResult.demo.llm_error).trim()}</span>
                  </li>
                ) : null}
              </ul>
            </>
          ) : (
            <p className="multitool-empty">{t("multitool_no_items")}</p>
          )}
        </div>

        <div className="multitool-card">
          <h3>{t("graph_visualization")}</h3>
          <div className="personal-tree-mini-window">{graphElement}</div>
          <div className="row-actions">
            <button type="button" disabled={busy || !demoResult} onClick={onRunPersonalTreeIngest}>
              {t("personal_tree_ingest_action")}
            </button>
            <button type="button" disabled={busy || !demoResult} onClick={onRefreshDrafts}>
              {t("action_refresh")}
            </button>
          </div>
        </div>

        {executionPanel}
        {reviewPanel}
      </div>
    </section>
  );
}

export function DailyOverviewSection({
  t,
  busy,
  interactionModePanel,
  dailyJournalText,
  onDailyJournalChange,
  onRunDailyMode,
  dailyModeResult,
  dailyOverallScore,
  dailyRecommendations,
  dailySignalChart,
  dailyScoreChart,
  graphElement,
  onRunPersonalTreeIngest,
  onRefreshDrafts,
  requestDraft,
  onRequestPatch,
  onSaveRequest,
  executionPanel,
  reviewPanel,
}) {
  return (
    <section className="card grid-2">
      <div>
        {interactionModePanel}
        <h2>{t("daily_mode")}</h2>
        <div className="row">
          <label>{t("daily_journal")}</label>
          <textarea
            value={dailyJournalText}
            onChange={(e) => onDailyJournalChange(e.target.value)}
            rows={8}
            placeholder={t("daily_journal_placeholder")}
          />
        </div>
        <div className="row-actions">
          <button disabled={busy} onClick={onRunDailyMode}>
            {t("run_daily_analysis")}
          </button>
        </div>
      </div>
      <div className="multitool-layout">
        <div className="multitool-dashboard">
          <h2>{t("daily_recommendations_scores")}</h2>
          {dailyModeResult ? (
            <>
              <div className="insight-metrics-grid">
                <div className="insight-metric-card">
                  <strong>overall</strong>
                  <span>{dailyOverallScore || 0}</span>
                </div>
                <div className="insight-metric-card">
                  <strong>{t("reasoning_chain")}</strong>
                  <span>{dailyRecommendations.length}</span>
                </div>
              </div>
              <div className="insight-chart-grid">
                <div>
                  <h4>{t("daily_mode")}</h4>
                  {dailySignalChart}
                </div>
                <div>
                  <h4>{t("daily_recommendations_scores")}</h4>
                  {dailyScoreChart}
                </div>
              </div>
              <div>
                <h4>{t("daily_recommendations_scores")}</h4>
                <ul className="insight-list">
                  {dailyRecommendations.map((item, index) => (
                    <li key={`daily-rec-${item?.id || index}`}>
                      <strong>{String(item?.title || `#${index + 1}`)}</strong>
                      <span>{String(item?.advice || "").trim() || "-"}</span>
                      <span>{String(item?.rationale || "").trim() || "-"}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </>
          ) : (
            <p className="multitool-empty">{t("multitool_no_items")}</p>
          )}
        </div>

        <div className="multitool-card">
          <h3>{t("graph_visualization")}</h3>
          <div className="personal-tree-mini-window">{graphElement}</div>
          <div className="row-actions">
            <button type="button" disabled={busy} onClick={onRunPersonalTreeIngest}>
              {t("personal_tree_ingest_action")}
            </button>
            <button type="button" disabled={busy} onClick={onRefreshDrafts}>
              {t("action_refresh")}
            </button>
          </div>
        </div>

        {executionPanel}
        {reviewPanel}

        <div className="multitool-card">
          <h3>{t("multitool_section_requests")}</h3>
          <div className="row">
            <label>{t("multitool_request_title")}</label>
            <input
              value={requestDraft.title}
              onChange={(e) => onRequestPatch({ title: e.target.value })}
            />
          </div>
          <div className="row">
            <label>{t("multitool_request_details")}</label>
            <textarea
              value={requestDraft.details}
              onChange={(e) => onRequestPatch({ details: e.target.value })}
              rows={4}
            />
          </div>
          <div className="row">
            <label>{t("multitool_request_output")}</label>
            <textarea
              value={requestDraft.desired_output}
              onChange={(e) => onRequestPatch({ desired_output: e.target.value })}
              rows={3}
            />
          </div>
          <div className="row-actions">
            <button type="button" disabled={busy} onClick={onSaveRequest}>
              {t("multitool_save")}
            </button>
            <button type="button" disabled={busy} onClick={onRefreshDrafts}>
              {t("action_refresh")}
            </button>
          </div>
        </div>
      </div>
    </section>
  );
}

export function UserGraphOverviewSection({
  t,
  busy,
  interactionModePanel,
  personalizationStudio,
  userForm,
  onApplyUserGraph,
  summaryPanel,
  reviewPanel,
}) {
  return (
    <section className="card grid-2">
      <div>
        {interactionModePanel}
        {personalizationStudio}
        {userForm}
        <div className="row-actions">
          <button disabled={busy} onClick={onApplyUserGraph}>
            {t("apply_user_graph")}
          </button>
        </div>
      </div>
      <div>
        {summaryPanel}
        {reviewPanel}
      </div>
    </section>
  );
}

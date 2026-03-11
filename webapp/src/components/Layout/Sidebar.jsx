import React from "react";

function SidebarButton({ active, children, onClick }) {
  return (
    <button type="button" className={`side-nav__button ${active ? "active" : ""}`} onClick={onClick}>
      {children}
    </button>
  );
}

export function Sidebar({
  workspace,
  onWorkspaceChange,
  sessions,
  activeSessionId,
  onSelectSession,
  onCreateSession,
  sessionTools,
  onSessionToolsChange,
  onLearnStyle,
  styleLearning,
  styleProfile,
  graphQuery,
  onGraphQueryChange,
  onUseSessionQuery,
  onQuickLoadHuman,
  onQuickLoadPsychology,
  seedRunning,
  t,
}) {
  return (
    <aside className="or-sidebar glass-panel session-sidebar">
      <section className="sidebar-block">
        <p className="sidebar-title">{t("sidebar_primary_view")}</p>
        <div className="side-nav inline">
          <SidebarButton active={workspace === "chat"} onClick={() => onWorkspaceChange("chat")}>{t("sidebar_chat_workspace")}</SidebarButton>
          <SidebarButton active={workspace === "graph"} onClick={() => onWorkspaceChange("graph")}>{t("sidebar_graph_workspace")}</SidebarButton>
        </div>
      </section>

      <section className="sidebar-block">
        <div className="sidebar-headline">
          <p className="sidebar-title">{t("sidebar_sessions")}</p>
          <button type="button" className="link-button" onClick={onCreateSession}>{t("sidebar_new_session")}</button>
        </div>
        <div className="queue-list compact session-list">
          {sessions.length ? (
            sessions.map((item) => (
              <button
                key={item.session_id}
                type="button"
                className={`session-card ${activeSessionId === item.session_id ? "active" : ""}`}
                onClick={() => onSelectSession(item.session_id)}
              >
                <strong>{item.title || t("sidebar_untitled_session")}</strong>
                <span>{item.updated_at || item.created_at || ""}</span>
                <p>{item.last_query || t("sidebar_session_empty")}</p>
              </button>
            ))
          ) : (
            <div className="empty-inline">{t("sidebar_no_sessions")}</div>
          )}
        </div>
      </section>

      {workspace === "chat" ? (
        <section className="sidebar-block">
          <p className="sidebar-title">{t("sidebar_tools")}</p>
          <div className="panel-form compact-grid">
            <label className="field-stack compact">
              <span>{t("chat_user_id")}</span>
              <input
                value={sessionTools.user_id || ""}
                onChange={(event) => onSessionToolsChange({ ...sessionTools, user_id: event.target.value })}
                placeholder={t("chat_user_id_placeholder")}
              />
            </label>
            <label className="field-stack compact span-2">
              <span>{t("chat_context")}</span>
              <textarea
                value={sessionTools.extra_context || ""}
                onChange={(event) => onSessionToolsChange({ ...sessionTools, extra_context: event.target.value })}
                placeholder={t("chat_context_placeholder")}
              />
            </label>
          </div>
          <div className="header-actions">
            <button type="button" className="button-secondary" onClick={onLearnStyle} disabled={styleLearning}>
              {styleLearning ? t("chat_learning_style") : t("chat_learn_style")}
            </button>
          </div>
          {styleProfile ? (
            <div className="sidebar-inline-note">
              <strong>{t("chat_style_profile")}</strong>
              <span>{t("chat_style_samples")}: {styleProfile.sample_count || 0}</span>
            </div>
          ) : null}
        </section>
      ) : (
        <section className="sidebar-block">
          <p className="sidebar-title">{t("sidebar_graph_tools")}</p>
          <div className="panel-form compact-grid">
            <label className="field-stack compact span-2">
              <span>{t("graph_query_label")}</span>
              <input
                value={graphQuery}
                onChange={(event) => onGraphQueryChange(event.target.value)}
                placeholder={t("graph_query_placeholder")}
              />
            </label>
          </div>
          <div className="header-actions">
            <button type="button" className="button-secondary" onClick={onUseSessionQuery}>{t("graph_use_session_query")}</button>
          </div>
          <div className="side-nav">
            <button type="button" className="side-nav__button" onClick={onQuickLoadPsychology} disabled={seedRunning}>{t("sidebar_psychology_base")}</button>
            <button type="button" className="side-nav__button" onClick={onQuickLoadHuman} disabled={seedRunning}>{t("sidebar_human_base")}</button>
          </div>
        </section>
      )}
    </aside>
  );
}

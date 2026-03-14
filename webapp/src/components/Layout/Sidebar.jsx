import React from 'react';

function SidebarButton({ active, children, onClick }) {
  return (
    <button type="button" className={`side-nav__button ${active ? 'active' : ''}`} onClick={onClick}>
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
  personalities,
  selectedPersonality,
  onSelectPersonality,
  onUploadFiles,
  uploadingFiles,
  graphQuery,
  onGraphQueryChange,
  onGraphSearch,
  onGraphRebuild,
  rebuildingGraph,
  t,
}) {
  return (
    <aside className="or-sidebar glass-panel session-sidebar">
      <section className="sidebar-block">
        <p className="sidebar-title">{t('sidebar_primary_view')}</p>
        <div className="side-nav inline">
          <SidebarButton active={workspace === 'chat'} onClick={() => onWorkspaceChange('chat')}>{t('sidebar_chat_workspace')}</SidebarButton>
          <SidebarButton active={workspace === 'graph'} onClick={() => onWorkspaceChange('graph')}>{t('sidebar_graph_workspace')}</SidebarButton>
        </div>
      </section>

      <section className="sidebar-block">
        <div className="sidebar-headline">
          <p className="sidebar-title">{t('sidebar_sessions')}</p>
          <button type="button" className="link-button" onClick={onCreateSession}>{t('sidebar_new_session')}</button>
        </div>
        <div className="queue-list compact session-list">
          {sessions.length ? sessions.map((item) => (
            <button
              key={item.session_id}
              type="button"
              className={`session-card ${activeSessionId === item.session_id ? 'active' : ''}`}
              onClick={() => onSelectSession(item.session_id)}
            >
              <strong>{item.title || t('sidebar_untitled_session')}</strong>
              <span>{item.updated_at || ''}</span>
              <p>{(item.messages || []).slice(-1)[0]?.message || t('sidebar_session_empty')}</p>
            </button>
          )) : <div className="empty-inline">{t('sidebar_no_sessions')}</div>}
        </div>
      </section>

      <section className="sidebar-block">
        <p className="sidebar-title">{t('sidebar_tools')}</p>
        <div className="panel-form compact-grid">
          <label className="field-stack compact span-2">
            <span>{t('chat_personality_select')}</span>
            <select value={selectedPersonality || ''} onChange={(event) => onSelectPersonality(event.target.value)}>
              <option value="">{t('chat_personality_none')}</option>
              {personalities.map((item) => (
                <option key={item.name} value={item.name}>{item.profile?.name || item.name}</option>
              ))}
            </select>
          </label>
          <label className="field-stack compact span-2">
            <span>{t('files_upload')}</span>
            <input type="file" accept=".txt,.md,.json,.csv" multiple onChange={(event) => onUploadFiles(event.target.files)} />
          </label>
        </div>
        <div className="header-actions">
          <span className="subtle-inline">{uploadingFiles ? t('files_uploading') : t('files_upload_hint')}</span>
        </div>
      </section>

      {workspace === 'graph' ? (
        <section className="sidebar-block">
          <p className="sidebar-title">{t('sidebar_graph_tools')}</p>
          <div className="panel-form compact-grid">
            <label className="field-stack compact span-2">
              <span>{t('graph_query')}</span>
              <input value={graphQuery} onChange={(event) => onGraphQueryChange(event.target.value)} placeholder={t('graph_query_placeholder')} />
            </label>
          </div>
          <div className="header-actions">
            <button type="button" className="button-secondary" onClick={onGraphSearch}>{t('graph_search')}</button>
            <button type="button" className="button-secondary" onClick={onGraphRebuild} disabled={rebuildingGraph}>
              {rebuildingGraph ? t('graph_rebuilding') : t('graph_rebuild')}
            </button>
          </div>
        </section>
      ) : null}
    </aside>
  );
}

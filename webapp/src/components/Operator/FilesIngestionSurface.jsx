import React from 'react';

export function FilesIngestionSurface({
  activeSessionId,
  uploadingFiles,
  onUploadFiles,
  lastUploadResult,
  t,
}) {
  return (
    <div className="operator-surface-grid files-surface-grid">
      <section className="workspace-panel glass-panel">
        <header className="panel-heading compact">
          <div>
            <p className="eyebrow">{t('files_surface_title')}</p>
            <h2>{t('files_surface_upload')}</h2>
          </div>
        </header>
        <div className="operator-stack">
          <div className="upload-dropzone">
            <strong>{t('files_upload')}</strong>
            <p>{t('files_upload_hint')}</p>
            <label className="button-secondary upload-button-inline">
              <input type="file" accept=".txt,.md,.json,.csv" multiple onChange={(event) => onUploadFiles(event.target.files)} />
              <span>{uploadingFiles ? t('files_uploading') : t('files_surface_choose')}</span>
            </label>
          </div>
          <div className="operator-kv-grid">
            <div><span>{t('files_surface_session')}</span><strong>{activeSessionId || '—'}</strong></div>
            <div><span>{t('files_surface_mode')}</span><strong>{'ingest -> validate -> graph merge'}</strong></div>
          </div>
        </div>
      </section>

      <section className="workspace-panel glass-panel operator-inspector-panel">
        <header className="panel-heading compact">
          <div>
            <p className="eyebrow">{t('files_surface_last')}</p>
            <h2>{t('files_surface_results')}</h2>
          </div>
        </header>
        {lastUploadResult?.files?.length ? (
          <ul className="dense-list">
            {lastUploadResult.files.map((item, index) => (
              <li key={`${item.path || 'file'}-${index}`}>
                <strong>{item.path?.split('/').slice(-1)[0] || `file-${index + 1}`}</strong>
                {' '}
                {item.result?.ok === false ? 'failed' : 'ingested'}
                {' '}
                {item.result?.reason || ''}
              </li>
            ))}
          </ul>
        ) : (
          <p className="empty-inline">{t('files_surface_empty')}</p>
        )}
      </section>
    </div>
  );
}

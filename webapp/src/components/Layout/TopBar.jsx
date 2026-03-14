import React from 'react';

function statusTone(ok) {
  return ok ? 'ok' : 'warn';
}

function Chip({ label, value, tone = 'neutral' }) {
  return (
    <div className={`status-chip ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export function TopBar({
  health,
  language,
  languageOptions,
  onLanguageChange,
  onRefresh,
  refreshing,
  t,
}) {
  return (
    <header className="or-topbar glass-panel">
      <div className="or-topbar__identity">
        <p className="eyebrow">{t('top_identity')}</p>
        <h1>{t('top_title')}</h1>
        <p className="subtext">{t('top_subtext')}</p>
      </div>
      <div className="or-topbar__actions">
        <div className="topbar-language">
          <span>{t('top_lang')}</span>
          <div className="pill-row compact">
            {languageOptions.map((item) => (
              <button
                key={item.code}
                type="button"
                className={`button-secondary chat-tab-button ${language === item.code ? 'active' : ''}`}
                onClick={() => onLanguageChange(item.code)}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>
        <div className="topbar-buttons">
          <button type="button" onClick={onRefresh} disabled={refreshing}>
            {refreshing ? t('top_refreshing') : t('top_refresh')}
          </button>
        </div>
      </div>
      <div className="or-topbar__status">
        <Chip label={t('top_workspace')} value={health?.ok ? t('status_online') : t('status_degraded')} tone={statusTone(Boolean(health?.ok))} />
      </div>
    </header>
  );
}

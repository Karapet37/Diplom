import React from "react";

function statusTone(ok, neutral = false) {
  if (neutral) {
    return "neutral";
  }
  return ok ? "ok" : "warn";
}

function Chip({ label, value, tone = "neutral" }) {
  return (
    <div className={`status-chip ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export function TopBar({
  health,
  engineHealth,
  controlState,
  searchTerm,
  onSearchChange,
  language,
  languageOptions,
  onLanguageChange,
  onRefresh,
  onRebuild,
  onRunQuality,
  refreshing,
  rebuilding,
  qualityRunning,
  t,
}) {
  const flags = controlState?.flags || {};
  const controllerMode = flags.read_only ? t("status_read_only") : flags.allow_graph_writes ? t("status_live") : t("status_locked");
  const writesAllowed = !flags.read_only && flags.allow_graph_writes;
  return (
    <header className="or-topbar glass-panel">
      <div className="or-topbar__identity">
        <p className="eyebrow">{t("top_identity")}</p>
        <h1>{t("top_title")}</h1>
        <p className="subtext">
          {t("top_subtext")}
        </p>
      </div>
      <div className="or-topbar__actions">
        <label className="or-search">
          <span>{t("top_search")}</span>
          <input
            type="text"
            value={searchTerm}
            onChange={(event) => onSearchChange(event.target.value)}
            placeholder={t("top_search_placeholder")}
          />
        </label>
        <div className="topbar-language">
          <span>{t("top_lang")}</span>
          <div className="pill-row compact">
            {languageOptions.map((item) => (
              <button
                key={item.code}
                type="button"
                className={`button-secondary chat-tab-button ${language === item.code ? "active" : ""}`}
                onClick={() => onLanguageChange(item.code)}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>
        <div className="topbar-buttons">
          {typeof onRunQuality === "function" ? (
            <button type="button" className="button-secondary" onClick={onRunQuality} disabled={qualityRunning}>
              {qualityRunning ? t("top_integrity_running") : t("top_integrity")}
            </button>
          ) : null}
          {typeof onRebuild === "function" ? (
            <button type="button" className="button-secondary" onClick={onRebuild} disabled={rebuilding}>
              {rebuilding ? t("top_rebuilding") : t("top_rebuild")}
            </button>
          ) : null}
          {typeof onRefresh === "function" ? (
            <button type="button" onClick={onRefresh} disabled={refreshing}>
              {refreshing ? t("top_refreshing") : t("top_refresh")}
            </button>
          ) : null}
        </div>
      </div>
      <div className="or-topbar__status">
        <Chip label={t("top_workspace")} value={health?.ok ? t("status_online") : t("status_degraded")} tone={statusTone(Boolean(health?.ok))} />
        <Chip label={t("top_graph_body")} value={engineHealth?.ok ? t("status_online") : t("status_degraded")} tone={statusTone(Boolean(engineHealth?.ok))} />
        <Chip label={t("top_controller")} value={controllerMode} tone={statusTone(writesAllowed, !flags.read_only && !flags.allow_graph_writes)} />
        <Chip label={t("top_writes")} value={writesAllowed ? t("status_allowed") : t("status_blocked")} tone={writesAllowed ? "ok" : "warn"} />
      </div>
    </header>
  );
}

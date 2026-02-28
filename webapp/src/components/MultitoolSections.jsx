import React from "react";

import { ResultSummaryPanel } from "./ProjectResultPanels";

function safeAttrs(node) {
  if (node?.attributes && typeof node.attributes === "object" && !Array.isArray(node.attributes)) {
    return node.attributes;
  }
  return {};
}

export function MultitoolEntityEditorCard({
  title,
  selectLabel,
  selectedId,
  onSelectId,
  nodes,
  optionLabel,
  draft,
  fields,
  onPatchDraft,
  onSave,
  onReset,
  busy = false,
  saveLabel = "Save",
  resetLabel = "New",
  resultTitle,
  result,
  emptyLabel = "No data yet.",
}) {
  return (
    <article className="multitool-card">
      <h3>{title}</h3>
      <div className="row">
        <label>{selectLabel}</label>
        <select value={String(selectedId || 0)} onChange={(event) => onSelectId(Number(event.target.value || 0))}>
          <option value="0">-</option>
          {(Array.isArray(nodes) ? nodes : []).map((node) => (
            <option key={`${title}-${node.id}`} value={String(node.id)}>
              {typeof optionLabel === "function"
                ? optionLabel(node)
                : `${node.id} · ${String(safeAttrs(node).title || safeAttrs(node).name || `#${node.id}`)}`}
            </option>
          ))}
        </select>
      </div>
      {(Array.isArray(fields) ? fields : []).map((field) => {
        const key = String(field?.key || "");
        if (!key) {
          return null;
        }
        const value = draft && typeof draft === "object" ? draft[key] ?? "" : "";
        const type = String(field?.type || "input");
        const onChange = (event) => onPatchDraft({ [key]: event.target.value });
        if (type === "textarea") {
          return (
            <div className="row" key={`${title}-${key}`}>
              <label>{field.label}</label>
              <textarea
                rows={Number(field?.rows || 3)}
                value={value}
                onChange={onChange}
                placeholder={field?.placeholder || undefined}
              />
            </div>
          );
        }
        if (type === "select") {
          return (
            <div className="row" key={`${title}-${key}`}>
              <label>{field.label}</label>
              <select value={value} onChange={onChange}>
                {(Array.isArray(field?.options) ? field.options : []).map((item) => {
                  const option =
                    item && typeof item === "object" && !Array.isArray(item)
                      ? item
                      : { value: item, label: String(item) };
                  return (
                    <option key={`${title}-${key}-${option.value}`} value={String(option.value)}>
                      {String(option.label ?? option.value)}
                    </option>
                  );
                })}
              </select>
            </div>
          );
        }
        return (
          <div className="row" key={`${title}-${key}`}>
            <label>{field.label}</label>
            <input
              type={field?.inputType || "text"}
              value={value}
              onChange={onChange}
              list={field?.listId || undefined}
              placeholder={field?.placeholder || undefined}
            />
          </div>
        );
      })}
      <div className="row-actions">
        <button type="button" disabled={busy} onClick={onSave}>
          {saveLabel}
        </button>
        <button type="button" onClick={onReset}>
          {resetLabel}
        </button>
      </div>
      <ResultSummaryPanel title={resultTitle} result={result} emptyLabel={emptyLabel} />
    </article>
  );
}

export function MultitoolDashboardSection({
  t,
  renderMiniBars,
  taskStatusRows,
  taskPriorityRows,
  riskProbabilityRows,
  riskImpactRows,
  domainCoverageRows,
  openTasks,
  topRisks,
  contradictionTopIssues,
  qualityTrendRows,
  backupHistoryRows,
  formatEpochLabel,
}) {
  return (
    <div className="multitool-dashboard">
      <h3>{t("multitool_dashboard")}</h3>
      <div className="multitool-stats-grid">
        <div>
          <h4>{t("multitool_chart_task_status")}</h4>
          {taskStatusRows.length ? renderMiniBars(taskStatusRows, "multitool-task-status") : <p className="multitool-empty">{t("multitool_no_items")}</p>}
        </div>
        <div>
          <h4>{t("multitool_chart_task_priority")}</h4>
          {taskPriorityRows.length ? renderMiniBars(taskPriorityRows, "multitool-task-priority") : <p className="multitool-empty">{t("multitool_no_items")}</p>}
        </div>
        <div>
          <h4>{t("multitool_chart_risk_probability")}</h4>
          {riskProbabilityRows.length ? renderMiniBars(riskProbabilityRows, "multitool-risk-prob") : <p className="multitool-empty">{t("multitool_no_items")}</p>}
        </div>
        <div>
          <h4>{t("multitool_chart_risk_impact")}</h4>
          {riskImpactRows.length ? renderMiniBars(riskImpactRows, "multitool-risk-impact") : <p className="multitool-empty">{t("multitool_no_items")}</p>}
        </div>
      </div>
      <div className="multitool-stats-grid">
        <div>
          <h4>{t("multitool_chart_domain_coverage")}</h4>
          {domainCoverageRows.length ? renderMiniBars(domainCoverageRows, "multitool-domain") : <p className="multitool-empty">{t("multitool_no_items")}</p>}
        </div>
        <div>
          <h4>{t("multitool_open_tasks")}</h4>
          <ul className="insight-list">
            {openTasks.length ? (
              openTasks.slice(0, 6).map((node) => {
                const attrs = safeAttrs(node);
                return (
                  <li key={`open-task-${node.id}`}>
                    <strong>{String(attrs.title || attrs.name || `#${node.id}`)}</strong>
                    <span>{`${attrs.status || "backlog"} · ${attrs.priority || "medium"}`}</span>
                  </li>
                );
              })
            ) : (
              <li>
                <span>{t("multitool_no_items")}</span>
              </li>
            )}
          </ul>
        </div>
        <div>
          <h4>{t("multitool_top_risks")}</h4>
          <ul className="insight-list">
            {topRisks.length ? (
              topRisks.map((node) => {
                const attrs = safeAttrs(node);
                return (
                  <li key={`top-risk-${node.id}`}>
                    <strong>{String(attrs.title || attrs.name || `#${node.id}`)}</strong>
                    <span>{`${attrs.probability || "medium"} · ${attrs.impact || "medium"}`}</span>
                  </li>
                );
              })
            ) : (
              <li>
                <span>{t("multitool_no_items")}</span>
              </li>
            )}
          </ul>
        </div>
      </div>
      <div className="multitool-stats-grid">
        <div>
          <h4>{t("multitool_widget_contradictions")}</h4>
          <ul className="insight-list">
            {contradictionTopIssues.length ? (
              contradictionTopIssues.map((row, index) => (
                <li key={`contr-top-${index}`}>
                  <strong>{String(row?.left_preview || `#${row?.left_node_id || index + 1}`)}</strong>
                  <span>{String(row?.right_preview || `#${row?.right_node_id || index + 1}`)}</span>
                  <span>{`score ${(Number(row?.score || 0) || 0).toFixed(2)}`}</span>
                </li>
              ))
            ) : (
              <li>
                <span>{t("multitool_no_items")}</span>
              </li>
            )}
          </ul>
        </div>
        <div>
          <h4>{t("multitool_widget_quality_trend")}</h4>
          {qualityTrendRows.length ? renderMiniBars(qualityTrendRows, "quality-trend") : <p className="multitool-empty">{t("multitool_no_items")}</p>}
        </div>
        <div>
          <h4>{t("multitool_widget_backup_history")}</h4>
          <ul className="insight-list">
            {backupHistoryRows.length ? (
              backupHistoryRows.map((row, index) => {
                const path = String(row?.path || "");
                const name = path ? path.split("/").pop() : `backup-${index + 1}`;
                return (
                  <li key={`backup-history-${index}`}>
                    <strong>{name}</strong>
                    <span>{formatEpochLabel(row?.modified_at)}</span>
                    <span>{`${Number(row?.size_bytes || 0)} B`}</span>
                  </li>
                );
              })
            ) : (
              <li>
                <span>{t("multitool_no_items")}</span>
              </li>
            )}
          </ul>
        </div>
      </div>
    </div>
  );
}

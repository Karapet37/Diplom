import React, { useEffect, useMemo, useState } from "react";

function asObject(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return {};
  }
  return value;
}

function asList(value) {
  if (Array.isArray(value)) {
    return value;
  }
  return [];
}

function asInt(value, fallback = 0) {
  const num = Number(value);
  if (!Number.isFinite(num)) {
    return fallback;
  }
  return Math.trunc(num);
}

function asFloat(value, fallback = 0) {
  const num = Number(value);
  if (!Number.isFinite(num)) {
    return fallback;
  }
  return num;
}

function compactJson(value) {
  try {
    return JSON.stringify(value);
  } catch (_error) {
    return String(value ?? "");
  }
}

function humanizeLabel(value) {
  return String(value || "")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function summarizePrimitive(value) {
  if (typeof value === "string") {
    const clean = value.trim();
    if (!clean) {
      return "-";
    }
    return clean.length > 140 ? `${clean.slice(0, 140).trim()}...` : clean;
  }
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(2);
  }
  if (typeof value === "boolean") {
    return value ? "yes" : "no";
  }
  if (Array.isArray(value)) {
    return `${value.length} items`;
  }
  if (value && typeof value === "object") {
    return `${Object.keys(value).length} fields`;
  }
  return "-";
}

function topSummaryEntries(root, limit = 6) {
  const excluded = new Set([
    "snapshot",
    "metrics",
    "raw_output",
    "events",
    "nodes",
    "edges",
    "items",
  ]);
  const rows = [];
  for (const [key, value] of Object.entries(asObject(root))) {
    if (excluded.has(String(key))) {
      continue;
    }
    if (value === undefined || value === null) {
      continue;
    }
    rows.push([humanizeLabel(key), summarizePrimitive(value)]);
    if (rows.length >= limit) {
      break;
    }
  }
  return rows;
}

function bestEntityPreview(root) {
  const source = asObject(root);
  const candidates = ["node", "edge", "review", "note", "extraction", "summary", "policy", "ingest", "stats"];
  for (const key of candidates) {
    const value = source[key];
    if (value && typeof value === "object" && !Array.isArray(value)) {
      return {
        key,
        value: asObject(value),
      };
    }
  }
  return null;
}

function metricCard(label, value, key) {
  return (
    <div className="insight-metric-card" key={key}>
      <strong>{String(value ?? "-")}</strong>
      <span>{label}</span>
    </div>
  );
}

export function ExecutionStatusPanel({ execution, title = "Execution" }) {
  const root = asObject(execution);
  const extraction = asObject(root.input_extraction);
  const monitor = asObject(root.graph_monitor);
  const status = String(root.status || (root.persisted ? "persisted" : "in_memory")).trim() || "in_memory";
  const metrics = [
    metricCard("status", status, "status"),
    metricCard("updates", asInt(extraction.updates_count, 0), "updates"),
    metricCard("patches", asInt(monitor.patch_total, 0), "patches"),
    metricCard("issues", asInt(extraction.issue_count, 0), "issues"),
  ];

  const details = [
    ["Persisted", root.persisted ? "yes" : "no"],
    ["Verified", extraction.verified ? "yes" : "no"],
    ["Model", String(extraction.model_path || monitor.model_path || "-").trim() || "-"],
    ["Attached", extraction.graph_attached || monitor.attached ? "yes" : "no"],
  ];

  return (
    <article className="multitool-card project-panel">
      <div className="project-panel-header">
        <h3>{title}</h3>
      </div>
      <div className="insight-metrics-grid">{metrics}</div>
      <div className="project-key-lines">
        {String(extraction.summary || "").trim() ? (
          <div className="project-key-line">
            <strong>Summary</strong>
            <span>{String(extraction.summary).trim()}</span>
          </div>
        ) : null}
        {details.map(([label, value]) => (
          <div className="project-key-line" key={`${title}-${label}`}>
            <strong>{label}</strong>
            <span>{value}</span>
          </div>
        ))}
      </div>
    </article>
  );
}

export function UserGraphSummaryPanel({ t, result }) {
  const root = asObject(result);
  const summary = asObject(root.summary);
  const userProfile = asObject(root.user_profile);
  const feedback = asObject(root.feedback_summary);
  const dimensions = asObject(userProfile.dimensions);
  const dimensionNames = asList(summary.dimension_names).length
    ? asList(summary.dimension_names)
    : Object.keys(dimensions);

  const metrics = [
    metricCard("dimensions", asInt(summary.dimension_count, dimensionNames.length), "dimensions"),
    metricCard("feedback", asInt(summary.feedback_items, feedback.total || 0), "feedback"),
    metricCard("profile", asInt(summary.profile_node_id, userProfile.profile_node_id || 0), "profile"),
    metricCard("saved", root.persisted ? "yes" : "no", "saved"),
  ];

  return (
    <div className="project-panel-stack">
      <article className="multitool-card project-panel">
        <div className="project-panel-header">
          <h2>{t("user_graph_update_result")}</h2>
        </div>
        <div className="insight-metrics-grid">{metrics}</div>
        <div className="project-key-lines">
          <div className="project-key-line">
            <strong>User</strong>
            <span>{String(summary.display_name || userProfile.display_name || "-")}</span>
          </div>
          <div className="project-key-line">
            <strong>Personalization</strong>
            <span>{summary.personalization_applied || root.personalization_applied ? "on" : "off"}</span>
          </div>
          {String(root.client_profile_error || "").trim() ? (
            <div className="project-key-line">
              <strong>Client</strong>
              <span>{String(root.client_profile_error).trim()}</span>
            </div>
          ) : null}
        </div>
        {dimensionNames.length ? (
          <div className="project-chip-row">
            {dimensionNames.slice(0, 12).map((item) => (
              <span className="project-chip" key={`dim-${item}`}>
                {String(item)}
              </span>
            ))}
          </div>
        ) : (
          <p className="project-muted">{t("multitool_no_items")}</p>
        )}
      </article>
      <ExecutionStatusPanel execution={root.execution || root} title="Input Pipeline" />
    </div>
  );
}

export function IntegrationLayerResultPanel({
  t,
  manifest,
  invokeResult,
  embedPreview,
  selectedAction,
}) {
  const manifestRoot = asObject(manifest);
  const invokeRoot = asObject(invokeResult);
  const manifestSummary = asObject(manifestRoot.summary);
  const invokeSummary = asObject(invokeRoot.summary);
  const actionSummary = asObject(invokeSummary.action_summary);
  const actions = asList(manifestRoot.actions);
  const preview = asObject(embedPreview);

  return (
    <div className="project-panel-stack">
      <article className="multitool-card project-panel">
        <div className="project-panel-header">
          <h3>{t("integration_layer_manifest")}</h3>
        </div>
        <div className="insight-metrics-grid">
          {metricCard("actions", asInt(manifestSummary.action_count, actions.length), "manifest-actions")}
          {metricCard("graph", asInt(manifestSummary.writes_graph_actions, 0), "manifest-graph")}
          {metricCard("chat", asInt(manifestSummary.chat_actions, 0), "manifest-chat")}
          {metricCard("models", asInt(manifestSummary.detected_model_count, 0), "manifest-models")}
        </div>
        <div className="project-key-lines">
          <div className="project-key-line">
            <strong>Host</strong>
            <span>{String(manifestRoot.host || preview.host || "-")}</span>
          </div>
          <div className="project-key-line">
            <strong>App</strong>
            <span>{String(manifestRoot.app_id || preview.app_id || "-")}</span>
          </div>
          {selectedAction?.description ? (
            <div className="project-key-line">
              <strong>Action</strong>
              <span>{String(selectedAction.description || "").trim()}</span>
            </div>
          ) : null}
        </div>
        {actions.length ? (
          <div className="project-chip-row">
            {actions.map((item) => (
              <span className="project-chip" key={`integration-action-${String(item?.key || "")}`}>
                {String(item?.key || "")}
              </span>
            ))}
          </div>
        ) : (
          <p className="project-muted">{t("multitool_no_items")}</p>
        )}
      </article>

      <article className="multitool-card project-panel">
        <div className="project-panel-header">
          <h3>{t("integration_layer_result")}</h3>
        </div>
        <div className="insight-metrics-grid">
          {metricCard("action", String(invokeSummary.action || preview.action || "-"), "invoke-action")}
          {metricCard("structured", asList(invokeSummary.structured_keys).length, "invoke-structured")}
          {metricCard("saved", invokeRoot.execution?.persisted || invokeRoot.result?.persisted ? "yes" : "no", "invoke-saved")}
          {metricCard("reply", invokeSummary.chat_response_present ? "yes" : "no", "invoke-reply")}
        </div>
        <div className="project-key-lines">
          {String(invokeRoot.chat_response || "").trim() ? (
            <div className="project-key-line">
              <strong>Chat</strong>
              <span>{String(invokeRoot.chat_response).trim()}</span>
            </div>
          ) : null}
          {Object.keys(actionSummary).length ? (
            Object.entries(actionSummary)
              .slice(0, 4)
              .map(([key, value]) => (
                <div className="project-key-line" key={`action-summary-${key}`}>
                  <strong>{key}</strong>
                  <span>{typeof value === "object" ? JSON.stringify(value) : String(value)}</span>
                </div>
              ))
          ) : (
            <div className="project-key-line">
              <strong>Embed</strong>
              <span>{String(preview.action || "-")}</span>
            </div>
          )}
        </div>
      </article>

      <ExecutionStatusPanel execution={invokeRoot.execution || invokeRoot.result || null} title="Graph Write" />
    </div>
  );
}

export function FlowReviewPanel({
  title = "Review",
  result,
  busy = false,
  onApplyArchiveUpdates,
  onApplyGraphMonitorPatches,
  onPreviewChange,
}) {
  const root = asObject(result);
  const inputExtraction = asObject(root.input_extraction);
  const graphMonitor = asObject(root.graph_monitor);
  const [updatesDraft, setUpdatesDraft] = useState("[]");
  const [patchesDraft, setPatchesDraft] = useState('{"node_patches":[],"edge_patches":[]}');
  const [localError, setLocalError] = useState("");
  const [lastStatus, setLastStatus] = useState("");

  useEffect(() => {
    setUpdatesDraft(JSON.stringify(asList(inputExtraction.updates), null, 2));
    setPatchesDraft(
      JSON.stringify(
        {
          node_patches: asList(graphMonitor.node_patches),
          edge_patches: asList(graphMonitor.edge_patches),
        },
        null,
        2
      )
    );
    setLocalError("");
    setLastStatus("");
  }, [inputExtraction, graphMonitor]);

  const patchPreview = useMemo(() => {
    const nodePatches = asList(graphMonitor.node_patches);
    const edgePatches = asList(graphMonitor.edge_patches);
    return {
      nodeCount: nodePatches.length,
      edgeCount: edgePatches.length,
    };
  }, [graphMonitor]);
  const parsedUpdates = useMemo(() => {
    try {
      const rows = JSON.parse(String(updatesDraft || "[]"));
      return Array.isArray(rows) ? rows : [];
    } catch (_error) {
      return null;
    }
  }, [updatesDraft]);
  const parsedPatchBundle = useMemo(() => {
    try {
      const parsed = asObject(JSON.parse(String(patchesDraft || "{}")));
      return {
        node_patches: asList(parsed.node_patches),
        edge_patches: asList(parsed.edge_patches),
      };
    } catch (_error) {
      return null;
    }
  }, [patchesDraft]);
  const graphPreview = useMemo(() => {
    const nodeIds = new Set();
    const edgeRefs = [];
    function addNodeCandidate(value) {
      const num = asInt(value, 0);
      if (num > 0) {
        nodeIds.add(num);
      }
    }
    function collectGraphRefs(item) {
      const row = asObject(item);
      addNodeCandidate(row.node_id);
      addNodeCandidate(row.target_node_id);
      addNodeCandidate(row.related_node_id);
      addNodeCandidate(row.left_node_id);
      addNodeCandidate(row.right_node_id);
      addNodeCandidate(row.from_node);
      addNodeCandidate(row.to_node);
      const fromNode = asInt(row.from_node, 0);
      const toNode = asInt(row.to_node, 0);
      if (fromNode > 0 && toNode > 0) {
        edgeRefs.push({
          from_node: fromNode,
          to_node: toNode,
          relation_type: String(row.relation_type || "").trim(),
          direction: String(row.direction || "").trim(),
          action: String(row.action || row.operation || "").trim(),
          weight: asFloat(row.weight, 0),
        });
      }
    }
    if (Array.isArray(parsedUpdates)) {
      parsedUpdates.forEach(collectGraphRefs);
    }
    if (parsedPatchBundle) {
      parsedPatchBundle.node_patches.forEach((row) => collectGraphRefs(row));
      parsedPatchBundle.edge_patches.forEach((row) => collectGraphRefs(row));
    }
    if (!nodeIds.size && !edgeRefs.length) {
      return null;
    }
    return {
      nodeIds: Array.from(nodeIds),
      edgeRefs,
    };
  }, [parsedPatchBundle, parsedUpdates]);
  const diffPreview = useMemo(() => {
    const snapshot = asObject(root.snapshot);
    const nodes = asList(snapshot.nodes);
    const edges = asList(snapshot.edges);
    const nodeById = new Map(nodes.map((row) => [asInt(row?.id, 0), asObject(row)]));

    const updateRows = Array.isArray(parsedUpdates)
      ? parsedUpdates.slice(0, 8).map((row, index) => {
          const item = asObject(row);
          return {
            id: `update-${index}`,
            label: `${String(item.operation || "upsert")} ${String(item.entity || "entity")}.${String(item.field || "field")}`,
            before: "graph review",
            after: summarizePrimitive(item.value),
            note: String(item.reason || "").trim(),
          };
        })
      : [];

    const nodeRows = parsedPatchBundle
      ? parsedPatchBundle.node_patches.slice(0, 8).map((row, index) => {
          const item = asObject(row);
          const node = nodeById.get(asInt(item.node_id, 0));
          const attrs = asObject(node?.attributes);
          const before = String(
            attrs.summary || attrs.monitor_summary || attrs.name || attrs.title || attrs.description || ""
          ).trim() || "-";
          const after = String(item.summary || "").trim() || "-";
          return {
            id: `node-patch-${index}`,
            label: `Node #${asInt(item.node_id, 0)}`,
            before,
            after,
            note: String(item.reason || "").trim(),
          };
        })
      : [];

    const edgeRows = parsedPatchBundle
      ? parsedPatchBundle.edge_patches.slice(0, 8).map((row, index) => {
          const item = asObject(row);
          const existing = edges.find((edge) => {
            const current = asObject(edge);
            return (
              asInt(current.from, 0) === asInt(item.from_node, 0) &&
              asInt(current.to, 0) === asInt(item.to_node, 0)
            );
          });
          const before = existing
            ? `${String(existing?.relation_type || "related_to")} @ ${asFloat(existing?.weight, 0).toFixed(2)}`
            : "none";
          const after = `${String(item.relation_type || "related_to")} @ ${asFloat(item.weight, 0).toFixed(2)}`;
          return {
            id: `edge-patch-${index}`,
            label: `Edge ${asInt(item.from_node, 0)} -> ${asInt(item.to_node, 0)}`,
            before,
            after,
            note: String(item.reason || item.action || "").trim(),
          };
        })
      : [];

    return { updateRows, nodeRows, edgeRows };
  }, [parsedUpdates, parsedPatchBundle, root.snapshot]);

  useEffect(() => {
    if (typeof onPreviewChange !== "function") {
      return undefined;
    }
    onPreviewChange(graphPreview);
    return () => {
      onPreviewChange(null);
    };
  }, [graphPreview, onPreviewChange]);

  async function applyUpdates() {
    try {
      const parsed = JSON.parse(String(updatesDraft || "[]"));
      if (!Array.isArray(parsed)) {
        throw new Error("updates draft must be a JSON array");
      }
      if (typeof onApplyArchiveUpdates === "function") {
        const out = await onApplyArchiveUpdates(parsed, root);
        const applied = asInt(out?.review?.archive_updates?.length, parsed.length);
        setLastStatus(`archive updates applied: ${applied}`);
      }
      setLocalError("");
    } catch (error) {
      setLocalError(String(error?.message || error));
    }
  }

  async function applyPatches() {
    try {
      const parsed = asObject(JSON.parse(String(patchesDraft || "{}")));
      const payload = {
        node_patches: asList(parsed.node_patches),
        edge_patches: asList(parsed.edge_patches),
      };
      if (typeof onApplyGraphMonitorPatches === "function") {
        const out = await onApplyGraphMonitorPatches(payload, root);
        setLastStatus(
          `monitor patches applied: ${asInt(out?.node_patch_count, payload.node_patches.length)} node / ${asInt(
            out?.edge_patch_count,
            payload.edge_patches.length
          )} edge`
        );
      }
      setLocalError("");
    } catch (error) {
      setLocalError(String(error?.message || error));
    }
  }

  return (
    <article className="multitool-card project-panel">
      <div className="project-panel-header">
        <h3>{title}</h3>
      </div>
      <div className="insight-metrics-grid">
        {metricCard("updates", asInt(inputExtraction.updates_count, asList(inputExtraction.updates).length), "review-updates")}
        {metricCard("node patches", patchPreview.nodeCount, "review-node-patches")}
        {metricCard("edge patches", patchPreview.edgeCount, "review-edge-patches")}
        {metricCard("verified", inputExtraction.verification?.verified ? "yes" : "no", "review-verified")}
      </div>
      <div className="project-key-lines">
        {(diffPreview.updateRows.length || diffPreview.nodeRows.length || diffPreview.edgeRows.length) ? (
          <div className="project-key-line">
            <strong>Diff Preview</strong>
            <div className="project-diff-grid">
              {diffPreview.updateRows.map((row) => (
                <div className="project-diff-row" key={row.id}>
                  <span className="project-diff-label">{row.label}</span>
                  <span className="project-diff-before">{row.before}</span>
                  <span className="project-diff-arrow">{"->"}</span>
                  <span className="project-diff-after">{row.after}</span>
                  {row.note ? <span className="project-diff-note">{row.note}</span> : null}
                </div>
              ))}
              {diffPreview.nodeRows.map((row) => (
                <div className="project-diff-row" key={row.id}>
                  <span className="project-diff-label">{row.label}</span>
                  <span className="project-diff-before">{row.before}</span>
                  <span className="project-diff-arrow">{"->"}</span>
                  <span className="project-diff-after">{row.after}</span>
                  {row.note ? <span className="project-diff-note">{row.note}</span> : null}
                </div>
              ))}
              {diffPreview.edgeRows.map((row) => (
                <div className="project-diff-row" key={row.id}>
                  <span className="project-diff-label">{row.label}</span>
                  <span className="project-diff-before">{row.before}</span>
                  <span className="project-diff-arrow">{"->"}</span>
                  <span className="project-diff-after">{row.after}</span>
                  {row.note ? <span className="project-diff-note">{row.note}</span> : null}
                </div>
              ))}
            </div>
          </div>
        ) : null}
        <div className="project-key-line">
          <strong>Archive Updates JSON</strong>
          <textarea
            rows={8}
            value={updatesDraft}
            onChange={(event) => setUpdatesDraft(event.target.value)}
          />
          {parsedUpdates === null ? <span className="project-error-text">Invalid updates JSON</span> : null}
        </div>
        <div className="project-key-line">
          <strong>Graph Monitor JSON</strong>
          <textarea
            rows={8}
            value={patchesDraft}
            onChange={(event) => setPatchesDraft(event.target.value)}
          />
          {parsedPatchBundle === null ? <span className="project-error-text">Invalid graph monitor JSON</span> : null}
        </div>
      </div>
      <div className="row-actions">
        <button type="button" disabled={busy} onClick={applyUpdates}>
          Confirm Updates
        </button>
        <button type="button" disabled={busy} onClick={applyPatches}>
          Apply Monitor
        </button>
      </div>
      {localError ? <p className="project-error-text">{localError}</p> : null}
      {lastStatus ? <p className="project-muted">{lastStatus}</p> : null}
    </article>
  );
}

export function ResultSummaryPanel({
  title,
  result,
  emptyLabel = "No result yet.",
}) {
  const root = asObject(result);
  const entity = bestEntityPreview(root);
  const summaryRows = topSummaryEntries(root, 6);
  if (!Object.keys(root).length) {
    return (
      <article className="multitool-card project-panel">
        <div className="project-panel-header">
          <h4>{title}</h4>
        </div>
        <p className="project-muted">{emptyLabel}</p>
      </article>
    );
  }

  const metrics = [
    metricCard("ok", root.ok || root.updated || root.created || root.deleted ? "yes" : "no", `${title}-ok`),
    metricCard("saved", root.persisted ? "yes" : "no", `${title}-saved`),
    metricCard("fields", Object.keys(root).length, `${title}-fields`),
    metricCard("entity", entity ? humanizeLabel(entity.key) : "-", `${title}-entity`),
  ];
  const entityRows = entity
    ? topSummaryEntries(entity.value, 4)
    : [];

  return (
    <article className="multitool-card project-panel">
      <div className="project-panel-header">
        <h4>{title}</h4>
      </div>
      <div className="insight-metrics-grid">{metrics}</div>
      <div className="project-key-lines">
        {summaryRows.map(([label, value]) => (
          <div className="project-key-line" key={`${title}-${label}`}>
            <strong>{label}</strong>
            <span>{value}</span>
          </div>
        ))}
        {entityRows.map(([label, value]) => (
          <div className="project-key-line" key={`${title}-entity-${label}`}>
            <strong>{`${humanizeLabel(entity?.key || "entity")} · ${label}`}</strong>
            <span>{value}</span>
          </div>
        ))}
      </div>
    </article>
  );
}

export function TextPreviewPanel({
  title,
  text,
  emptyLabel = "No data yet.",
}) {
  const value = typeof text === "string" ? text : text == null ? "" : compactJson(text);
  const clean = String(value || "").trim();
  if (!clean) {
    return (
      <article className="multitool-card project-panel">
        <div className="project-panel-header">
          <h4>{title}</h4>
        </div>
        <p className="project-muted">{emptyLabel}</p>
      </article>
    );
  }
  return (
    <article className="multitool-card project-panel">
      <div className="project-panel-header">
        <h4>{title}</h4>
      </div>
      <div className="insight-metrics-grid">
        {metricCard("chars", clean.length, `${title}-chars`)}
        {metricCard("lines", clean.split(/\r?\n/).length, `${title}-lines`)}
      </div>
      <textarea className="project-preview-text" readOnly rows={8} value={clean} />
    </article>
  );
}

export function ClientSummaryPanel({ t, profile }) {
  const root = asObject(profile);
  const browser = asObject(root.browser);
  const os = asObject(root.os);
  const network = asObject(root.network);
  const ip = asObject(network.ip);
  const locale = asObject(root.locale);
  const screen = asObject(root.screen);
  const viewport = asObject(root.viewport);

  return (
    <article className="card project-panel">
      <div className="project-panel-header">
        <h2>{t("client_profile_semantic_input")}</h2>
      </div>
      <div className="insight-metrics-grid">
        {metricCard("browser", String(browser.family || "-"), "client-browser")}
        {metricCard("os", String(os.family || "-"), "client-os")}
        {metricCard("lang", String(locale.language || "-"), "client-lang")}
        {metricCard("vpn", network.vpn_proxy_suspected ? "yes" : "no", "client-vpn")}
      </div>
      <div className="project-key-lines">
        <div className="project-key-line">
          <strong>IP</strong>
          <span>{String(ip.ip || "-")}</span>
        </div>
        <div className="project-key-line">
          <strong>Timezone</strong>
          <span>{String(locale.timezone || "-")}</span>
        </div>
        <div className="project-key-line">
          <strong>Screen</strong>
          <span>{`${asInt(screen.width, 0)} x ${asInt(screen.height, 0)} / ${asInt(viewport.width, 0)} x ${asInt(viewport.height, 0)}`}</span>
        </div>
        <div className="project-key-line">
          <strong>User Agent</strong>
          <span>{String(root.user_agent || browser.user_agent || "-")}</span>
        </div>
      </div>
    </article>
  );
}

export function AdvisorsSummaryPanel({
  t,
  debateResult,
  advisorDetectedModels,
  advisorRoleModels,
  archiveChatResult,
  archiveReviewResult,
  promptCatalog,
}) {
  const debate = asObject(debateResult);
  const decision = asObject(debate.decision);
  const hypotheses = asList(debate.hypotheses);
  const archiveVerification = asObject(asObject(archiveChatResult).verification);
  const archiveReview = asObject(archiveReviewResult);
  const prompts = asList(promptCatalog);

  return (
    <div className="project-panel-stack">
      <section className="card grid-2">
        <article className="multitool-card project-panel">
          <div className="project-panel-header">
            <h2>{t("debate_result")}</h2>
          </div>
          <div className="insight-metrics-grid">
            {metricCard("hypotheses", hypotheses.length, "debate-h")}
            {metricCard("selected", asInt(decision.selected_index, 0), "debate-sel")}
            {metricCard("confidence", asFloat(decision.confidence, 0).toFixed(2), "debate-conf")}
            {metricCard("saved", debate.persisted ? "yes" : "no", "debate-save")}
          </div>
          <div className="project-key-lines">
            <div className="project-key-line">
              <strong>Decision</strong>
              <span>{String(decision.decision || "-")}</span>
            </div>
            <div className="project-key-line">
              <strong>Consensus</strong>
              <span>{String(decision.consensus || "-")}</span>
            </div>
          </div>
          {hypotheses.length ? (
            <ul className="insight-list">
              {hypotheses.slice(0, 4).map((item, index) => (
                <li key={`hyp-${index}`}>
                  <strong>{String(item?.title || `#${index + 1}`)}</strong>
                  <span>{String(item?.claim || "").trim() || "-"}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="project-muted">{t("multitool_no_items")}</p>
          )}
        </article>

        <article className="multitool-card project-panel">
          <div className="project-panel-header">
            <h2>{t("mini_coders_advisors")}</h2>
          </div>
          <div className="insight-metrics-grid">
            {metricCard("models", asList(advisorDetectedModels).length, "advisor-models")}
            {metricCard("roles", asList(advisorRoleModels).length, "advisor-roles")}
            {metricCard("prompts", prompts.length, "advisor-prompts")}
            {metricCard("verified", archiveVerification.verified ? "yes" : "no", "advisor-verified")}
          </div>
          {asList(advisorDetectedModels).length ? (
            <div className="project-chip-row">
              {asList(advisorDetectedModels).map((item) => (
                <span className="project-chip" key={`detected-${item}`}>
                  {String(item)}
                </span>
              ))}
            </div>
          ) : null}
          <div className="project-key-lines">
            {asList(advisorRoleModels)
              .slice(0, 4)
              .map((item, index) => (
                <div className="project-key-line" key={`advisor-role-${index}`}>
                  <strong>{String(item?.role || `role_${index + 1}`)}</strong>
                  <span>{String(item?.selected_model_path || item?.model_path || "-")}</span>
                </div>
              ))}
          </div>
        </article>
      </section>

      <section className="card grid-2">
        <article className="multitool-card project-panel">
          <div className="project-panel-header">
            <h2>{t("archive_chat_result")}</h2>
          </div>
          <div className="insight-metrics-grid">
            {metricCard("verified", archiveVerification.verified ? "yes" : "no", "archive-verified")}
            {metricCard("issues", asInt(archiveVerification.issue_count, 0), "archive-issues")}
            {metricCard("warnings", asInt(archiveVerification.warning_count, 0), "archive-warnings")}
            {metricCard("score", asFloat(archiveVerification.score, 0).toFixed(2), "archive-score")}
          </div>
          <div className="project-key-lines">
            <div className="project-key-line">
              <strong>Summary</strong>
              <span>{String(asObject(archiveChatResult).summary || "-")}</span>
            </div>
            <div className="project-key-line">
              <strong>Review</strong>
              <span>{String(archiveReview.summary || "-")}</span>
            </div>
          </div>
        </article>

        <article className="multitool-card project-panel">
          <div className="project-panel-header">
            <h2>{t("prompt_catalog")}</h2>
          </div>
          {prompts.length ? (
            <div className="project-chip-row">
              {prompts.slice(0, 16).map((item, index) => (
                <span className="project-chip" key={`prompt-${item?.name || index}`}>
                  {String(item?.name || `prompt_${index + 1}`)}
                </span>
              ))}
            </div>
          ) : (
            <p className="project-muted">{t("multitool_no_items")}</p>
          )}
        </article>
      </section>
    </div>
  );
}

export function HallucinationSummaryPanel({ t, reportResult, checkResult }) {
  const report = asObject(reportResult);
  const check = asObject(checkResult);
  const verification = asObject(check.verification);

  return (
    <div className="project-panel-stack">
      <article className="multitool-card project-panel">
        <div className="project-panel-header">
          <h3>{t("hallucination_report_result")}</h3>
        </div>
        <div className="insight-metrics-grid">
          {metricCard("saved", report.persisted ? "yes" : "no", "hall-save")}
          {metricCard("severity", String(report.severity || "-"), "hall-severity")}
          {metricCard("matches", asInt(report.match_count, 0), "hall-matches")}
          {metricCard("memory", asInt(report.memory_hits, 0), "hall-memory")}
        </div>
        <div className="project-key-lines">
          <div className="project-key-line">
            <strong>Prompt</strong>
            <span>{String(report.prompt || "-")}</span>
          </div>
          <div className="project-key-line">
            <strong>Correct</strong>
            <span>{String(report.correct_answer || "-")}</span>
          </div>
        </div>
      </article>

      <article className="multitool-card project-panel">
        <div className="project-panel-header">
          <h3>{t("hallucination_check_result")}</h3>
        </div>
        <div className="insight-metrics-grid">
          {metricCard("verified", verification.verified ? "yes" : "no", "hall-check-verified")}
          {metricCard("issues", asInt(verification.issue_count, 0), "hall-check-issues")}
          {metricCard("warnings", asInt(verification.warning_count, 0), "hall-check-warnings")}
          {metricCard("score", asFloat(verification.score, 0).toFixed(2), "hall-check-score")}
        </div>
        <div className="project-key-lines">
          <div className="project-key-line">
            <strong>Decision</strong>
            <span>{String(check.status || "-")}</span>
          </div>
          <div className="project-key-line">
            <strong>Matches</strong>
            <span>{compactJson(asList(check.matches).slice(0, 4))}</span>
          </div>
        </div>
      </article>
    </div>
  );
}

export function DataSummaryPanels({
  t,
  eventsView,
  nodesView,
  edgesView,
  setEventsPage,
  setNodesPage,
  setEdgesPage,
  modulesView,
  setModulesPage,
  renderPager,
  dbSchema,
}) {
  const schema = asObject(dbSchema);
  const tables = asList(schema.tables);

  return (
    <>
      <section className="card grid-2">
        <div>
          <h2>{t("event_stream")}</h2>
          {renderPager({
            page: eventsView.page,
            totalPages: eventsView.totalPages,
            setPage: setEventsPage,
            label: t("pager_events"),
          })}
          {asList(eventsView.items).length ? (
            <ul className="insight-list">
              {asList(eventsView.items).map((item, index) => (
                <li key={`event-row-${index}`}>
                  <strong>{String(item?.type || item?.event_type || `event_${index + 1}`)}</strong>
                  <span>{String(item?.timestamp || item?.ts || "").trim() || "-"}</span>
                  <span>{compactJson(asObject(item?.payload || item?.data || item))}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="project-muted">{t("multitool_no_items")}</p>
          )}
        </div>
        <div>
          <h2>{t("snapshot_nodes")}</h2>
          {renderPager({
            page: nodesView.page,
            totalPages: nodesView.totalPages,
            setPage: setNodesPage,
            label: t("pager_nodes"),
          })}
          {asList(nodesView.items).length ? (
            <ul className="insight-list">
              {asList(nodesView.items).map((item, index) => (
                <li key={`node-row-${item?.id || index}`}>
                  <strong>{String(item?.id || index + 1)} · {String(item?.type || "node")}</strong>
                  <span>{String(item?.attributes?.name || item?.attributes?.title || item?.attributes?.summary || "-")}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="project-muted">{t("multitool_no_items")}</p>
          )}
        </div>
      </section>

      <section className="card grid-2">
        <div>
          <h2>{t("snapshot_edges")}</h2>
          {renderPager({
            page: edgesView.page,
            totalPages: edgesView.totalPages,
            setPage: setEdgesPage,
            label: t("pager_edges"),
          })}
          {asList(edgesView.items).length ? (
            <ul className="insight-list">
              {asList(edgesView.items).map((item, index) => (
                <li key={`edge-row-${index}`}>
                  <strong>{`${String(item?.from || "?")} -> ${String(item?.to || "?")}`}</strong>
                  <span>{String(item?.relation_type || "related_to")}</span>
                  <span>{asFloat(item?.weight, 0).toFixed(2)}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="project-muted">{t("multitool_no_items")}</p>
          )}
        </div>
        <article className="multitool-card project-panel">
          <div className="project-panel-header">
            <h2>{t("sql_table_schema_json")}</h2>
          </div>
          <div className="insight-metrics-grid">
            {metricCard("tables", tables.length, "schema-tables")}
            {metricCard("current", String(schema.schema_version || "-"), "schema-current")}
            {metricCard("target", String(schema.target_schema_version || "-"), "schema-target")}
            {metricCard("migrations", asList(schema.applied_versions).length, "schema-migrations")}
          </div>
          <div className="project-chip-row">
            {tables.slice(0, 12).map((item) => (
              <span className="project-chip" key={`schema-${String(item?.name || "")}`}>
                {String(item?.name || "")}
              </span>
            ))}
          </div>
        </article>
      </section>

      <section className="card">
        <h2>{t("project_modules")}</h2>
        {renderPager({
          page: modulesView.page,
          totalPages: modulesView.totalPages,
          setPage: setModulesPage,
          label: t("pager_modules"),
        })}
        <div className="modules-grid">
          {modulesView.items.map((mod) => (
            <article key={mod.name} className="module-card">
              <h3>{mod.name}</h3>
              <p>{mod.description}</p>
              <p>
                {t("files")}: {mod.count}
              </p>
              <details>
                <summary>{t("show_files")}</summary>
                <pre>{(mod.files || []).join("\n")}</pre>
              </details>
            </article>
          ))}
        </div>
      </section>
    </>
  );
}

import React, { useMemo } from "react";
import { getNodeExamples, getNodeExplanation, getNodeGloss, getNodeImages, getNodeLabel, getNodeTags } from "../../lib/graphView";

function shortenText(value, limit = 220) {
  const text = String(value || "").trim();
  if (!text) return "";
  return text.length <= limit ? text : `${text.slice(0, limit - 3).trimEnd()}...`;
}

function ListBlock({ title, items, renderItem, emptyText = "Nothing available." }) {
  return (
    <section className="inspector-block">
      <header><h3>{title}</h3></header>
      {items && items.length ? <ul className="dense-list">{items.map(renderItem)}</ul> : <div className="empty-inline">{emptyText}</div>}
    </section>
  );
}

function InspectorStat({ label, value }) {
  return <div className="inspector-stat"><span>{label}</span><strong>{value}</strong></div>;
}

export function InspectorPanel({ selection, graph, controlState, llmPolicy, qualityReport, onOpenEdit, t }) {
  const nodes = graph?.nodes || [];
  const edges = graph?.edges || [];

  const selectedNode = useMemo(() => selection?.kind === "node" ? nodes.find((node) => String(node.id) === String(selection.id)) || null : null, [selection, nodes]);
  const selectedEdge = useMemo(() => selection?.kind === "edge" ? edges.find((edge) => String(edge.edge_key || `${edge.src_id}|${edge.type}|${edge.dst_id}`) === String(selection.id)) || null : null, [selection, edges]);

  const nodeNeighbors = useMemo(() => {
    if (!selectedNode) return [];
    return edges
      .filter((edge) => String(edge.src_id) === String(selectedNode.id) || String(edge.dst_id) === String(selectedNode.id))
      .map((edge) => {
        const neighborId = String(edge.src_id) === String(selectedNode.id) ? edge.dst_id : edge.src_id;
        const neighbor = nodes.find((node) => String(node.id) === String(neighborId));
        return { edge, neighbor };
      });
  }, [selectedNode, edges, nodes]);

  const edgeEvidence = selectedEdge?.evidence || [];
  const nodeImages = selectedNode ? getNodeImages(selectedNode) : [];
  const nodeFindings = selectedNode ? (qualityReport?.node_findings_by_id?.[selectedNode.id] || []) : [];

  if (selectedNode) {
    return (
      <section className="workspace-panel glass-panel">
        <section className="inspector-hero">
          <p className="eyebrow">{t("inspector_node")}</p>
          <h2>{getNodeLabel(selectedNode)}</h2>
          <div className="inspector-pill-row">
            <span className="pill">{selectedNode.type}</span>
            <span className="pill">{nodeNeighbors.length} {t("inspector_connections")}</span>
          </div>
          <div className="inspector-actions">
            <button type="button" className="button-secondary" onClick={onOpenEdit}>{t("inspector_edit_node")}</button>
          </div>
        </section>
        <section className="inspector-block"><header><h3>{t("inspector_short_gloss")}</h3></header><p>{getNodeGloss(selectedNode) || t("inspector_no_gloss")}</p></section>
        <section className="inspector-block"><header><h3>{t("inspector_plain_explanation")}</h3></header><p>{getNodeExplanation(selectedNode) || t("inspector_no_explanation")}</p></section>
        {selectedNode.what_it_is ? <section className="inspector-block"><header><h3>{t("editor_what_it_is")}</h3></header><p>{selectedNode.what_it_is}</p></section> : null}
        {selectedNode.how_it_works ? <section className="inspector-block"><header><h3>{t("editor_how_it_works")}</h3></header><p>{selectedNode.how_it_works}</p></section> : null}
        {selectedNode.how_to_recognize ? <section className="inspector-block"><header><h3>{t("editor_how_to_recognize")}</h3></header><p>{selectedNode.how_to_recognize}</p></section> : null}
        {selectedNode.importance_vector ? (
          <section className="inspector-block grid-stats">
            <InspectorStat label={t("inspector_logic_weight")} value={Number(selectedNode.importance_vector.logic_weight || 0).toFixed(2)} />
            <InspectorStat label={t("inspector_emotion_weight")} value={Number(selectedNode.importance_vector.emotion_weight || 0).toFixed(2)} />
            <InspectorStat label={t("inspector_risk_weight")} value={Number(selectedNode.importance_vector.risk_weight || 0).toFixed(2)} />
            <InspectorStat label={t("inspector_relevance_weight")} value={Number(selectedNode.importance_vector.relevance_weight || 0).toFixed(2)} />
          </section>
        ) : null}
        <ListBlock title={t("inspector_examples")} items={getNodeExamples(selectedNode)} renderItem={(item, index) => <li key={`${item}-${index}`}>{String(item)}</li>} emptyText={t("inspector_no_examples")} />
        <ListBlock title={t("inspector_tags")} items={getNodeTags(selectedNode)} renderItem={(item, index) => <li key={`${item}-${index}`}><span className="pill">{String(item)}</span></li>} emptyText={t("inspector_no_tags")} />
        <ListBlock title={t("inspector_quality_findings")} items={nodeFindings} renderItem={(item, index) => <li key={`${item}-${index}`}>{String(item)}</li>} emptyText={t("inspector_no_findings")} />
        <ListBlock
          title={t("inspector_connected_nodes")}
          items={nodeNeighbors}
          renderItem={({ edge, neighbor }) => (
            <li key={`${edge.edge_key}`}>
              <strong>{neighbor ? getNodeLabel(neighbor) : edge.dst_id}</strong>
              <span>{edge.type}{neighbor?.type ? ` · ${neighbor.type}` : ""}</span>
              {neighbor ? <p>{shortenText(getNodeGloss(neighbor) || getNodeExplanation(neighbor), 180)}</p> : null}
            </li>
          )}
          emptyText={t("inspector_no_connected")}
        />
        <ListBlock
          title={t("inspector_evidence")}
          items={nodeNeighbors.flatMap((item) => item.edge.evidence || []).slice(0, 12)}
          renderItem={(item, index) => <li key={`${item.source_id}-${index}`}><strong>{item.source_id}</strong><span>{item.snippet_text}</span></li>}
          emptyText={t("inspector_no_evidence_connected")}
        />
        <ListBlock
          title={t("inspector_images")}
          items={nodeImages}
          renderItem={(item, index) => <li key={`${item}-${index}`}><img src={item} alt={getNodeLabel(selectedNode)} className="inspector-image" /></li>}
          emptyText={t("inspector_no_images")}
        />
      </section>
    );
  }

  if (selectedEdge) {
    return (
      <section className="workspace-panel glass-panel">
        <section className="inspector-hero">
          <p className="eyebrow">{t("inspector_relation")}</p>
          <h2>{selectedEdge.type}</h2>
          <div className="inspector-pill-row">
            <span className="pill">{selectedEdge.src_id}</span>
            <span className="pill">{selectedEdge.dst_id}</span>
          </div>
          <div className="inspector-actions"><button type="button" className="button-secondary" onClick={onOpenEdit}>{t("inspector_edit_relation")}</button></div>
        </section>
        <section className="inspector-block grid-stats">
          <InspectorStat label={t("inspector_weight")} value={Number(selectedEdge.weight || 0).toFixed(2)} />
          <InspectorStat label={t("inspector_confidence")} value={Number(selectedEdge.confidence || 0).toFixed(2)} />
          <InspectorStat label={t("inspector_evidence_count")} value={String(edgeEvidence.length)} />
        </section>
        <ListBlock title={t("inspector_evidence")} items={edgeEvidence} renderItem={(item, index) => <li key={`${item.source_id}-${index}`}><strong>{item.source_id}</strong><span>{item.snippet_text}</span></li>} emptyText={t("inspector_no_evidence_edge")} />
      </section>
    );
  }

  const flags = controlState?.flags || {};
  return (
    <section className="workspace-panel glass-panel">
      <section className="inspector-hero">
        <p className="eyebrow">{t("inspector")}</p>
        <h2>{t("inspector_system_overview")}</h2>
        <p>{t("inspector_select_hint")}</p>
      </section>
      <section className="inspector-block grid-stats">
        <InspectorStat label={t("inspector_read_only")} value={flags.read_only ? t("inspector_on") : t("inspector_off")} />
        <InspectorStat label={t("inspector_graph_writes")} value={flags.allow_graph_writes ? t("inspector_enabled") : t("inspector_disabled")} />
        <InspectorStat label={t("inspector_prompt_execution")} value={flags.allow_prompt_execution ? t("inspector_enabled") : t("inspector_disabled")} />
      </section>
      <section className="inspector-block"><header><h3>{t("inspector_policy")}</h3></header><p>{t("inspector_mode")}: <strong>{llmPolicy?.policy?.mode || llmPolicy?.mode || "unknown"}</strong></p></section>
    </section>
  );
}

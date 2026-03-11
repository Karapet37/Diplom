import React from "react";

const EDGE_TYPE_OPTIONS = [
  "RELATED_TO",
  "IS_PART_OF",
  "INSTANCE_OF",
  "EXAMPLE_OF",
  "HAS_TRAIT",
  "USES_PATTERN",
  "SAID_EXAMPLE",
  "WORKS_IN_DOMAIN",
  "RESPONDS_WITH",
];

export function GraphEditorPanel({
  selection,
  nodeDraft,
  edgeDraft,
  createNodeDraft,
  relationDraft,
  visibleNodes,
  onNodeDraftChange,
  onEdgeDraftChange,
  onCreateNodeDraftChange,
  onRelationDraftChange,
  onSaveNode,
  onSaveEdge,
  onCreateNode,
  onCreateRelation,
  savingNode,
  savingEdge,
  creatingNode,
  creatingRelation,
  t,
}) {
  const selectedNodeId = selection?.kind === "node" ? selection.id : "";

  return (
    <section className="workspace-panel glass-panel node-editor-shell">
      <header className="panel-heading">
        <div>
          <p className="eyebrow">{t("editor_eyebrow")}</p>
          <h2>{t("editor_title")}</h2>
          <p>{t("editor_subtitle")}</p>
        </div>
      </header>

      {selection?.kind === "node" ? (
        <section className="controller-card wide">
          <div className="panel-heading compact">
            <div>
              <h3>{t("editor_node_title")}</h3>
              <p>{t("editor_node_text")}</p>
            </div>
            <button type="button" onClick={onSaveNode} disabled={savingNode || !nodeDraft.id}>
              {savingNode ? t("editor_saving") : t("editor_save_node")}
            </button>
          </div>
          <div className="panel-form">
            <label className="field-stack compact">
              <span>{t("editor_node_id")}</span>
              <input value={nodeDraft.id || ""} readOnly disabled />
            </label>
            <label className="field-stack compact">
              <span>{t("editor_name")}</span>
              <input value={nodeDraft.label || ""} onChange={(event) => onNodeDraftChange({ ...nodeDraft, label: event.target.value })} />
            </label>
            <label className="field-stack compact">
              <span>{t("editor_type")}</span>
              <input value={nodeDraft.type || ""} onChange={(event) => onNodeDraftChange({ ...nodeDraft, type: event.target.value })} />
            </label>
            <label className="field-stack">
              <span>{t("editor_summary")}</span>
              <textarea value={nodeDraft.short_gloss || ""} onChange={(event) => onNodeDraftChange({ ...nodeDraft, short_gloss: event.target.value })} />
            </label>
            <label className="field-stack">
              <span>{t("editor_what_it_is")}</span>
              <textarea value={nodeDraft.what_it_is || ""} onChange={(event) => onNodeDraftChange({ ...nodeDraft, what_it_is: event.target.value })} />
            </label>
            <label className="field-stack">
              <span>{t("editor_how_it_works")}</span>
              <textarea value={nodeDraft.how_it_works || ""} onChange={(event) => onNodeDraftChange({ ...nodeDraft, how_it_works: event.target.value })} />
            </label>
            <label className="field-stack">
              <span>{t("editor_how_to_recognize")}</span>
              <textarea value={nodeDraft.how_to_recognize || ""} onChange={(event) => onNodeDraftChange({ ...nodeDraft, how_to_recognize: event.target.value })} />
            </label>
            <label className="field-stack">
              <span>{t("editor_examples")}</span>
              <textarea value={nodeDraft.examplesText || ""} onChange={(event) => onNodeDraftChange({ ...nodeDraft, examplesText: event.target.value })} />
            </label>
            <label className="field-stack">
              <span>{t("editor_tags")}</span>
              <textarea value={nodeDraft.tagsText || ""} onChange={(event) => onNodeDraftChange({ ...nodeDraft, tagsText: event.target.value })} />
            </label>
          </div>
        </section>
      ) : null}

      {selection?.kind === "edge" ? (
        <section className="controller-card wide">
          <div className="panel-heading compact">
            <div>
              <h3>{t("editor_edge_title")}</h3>
              <p>{t("editor_edge_text")}</p>
            </div>
            <button type="button" onClick={onSaveEdge} disabled={savingEdge || !edgeDraft.src_id}>
              {savingEdge ? t("editor_saving") : t("editor_save_edge")}
            </button>
          </div>
          <div className="panel-form grid-two">
            <label className="field-stack compact">
              <span>{t("editor_source")}</span>
              <input value={edgeDraft.src_id || ""} readOnly disabled />
            </label>
            <label className="field-stack compact">
              <span>{t("editor_target")}</span>
              <input value={edgeDraft.dst_id || ""} readOnly disabled />
            </label>
            <label className="field-stack compact span-2">
              <span>{t("editor_relation_type")}</span>
              <input value={edgeDraft.type || ""} readOnly disabled />
            </label>
            <label className="field-stack compact">
              <span>{t("inspector_weight")}</span>
              <input type="number" min="0" step="0.05" value={edgeDraft.weight ?? 1} onChange={(event) => onEdgeDraftChange({ ...edgeDraft, weight: Number(event.target.value || 0) })} />
            </label>
            <label className="field-stack compact">
              <span>{t("inspector_confidence")}</span>
              <input type="number" min="0" max="1" step="0.05" value={edgeDraft.confidence ?? 0.7} onChange={(event) => onEdgeDraftChange({ ...edgeDraft, confidence: Number(event.target.value || 0) })} />
            </label>
          </div>
        </section>
      ) : null}

      <section className="controller-card wide">
        <div className="panel-heading compact">
          <div>
            <h3>{t("editor_create_node_title")}</h3>
            <p>{t("editor_create_node_text")}</p>
          </div>
          <button type="button" className="button-secondary" onClick={onCreateNode} disabled={creatingNode || !createNodeDraft.node_id || !createNodeDraft.label}>
            {creatingNode ? t("editor_saving") : t("editor_create_node")}
          </button>
        </div>
        <div className="panel-form">
          <label className="field-stack compact">
            <span>{t("editor_node_id")}</span>
            <input value={createNodeDraft.node_id || ""} onChange={(event) => onCreateNodeDraftChange({ ...createNodeDraft, node_id: event.target.value })} />
          </label>
          <label className="field-stack compact">
            <span>{t("editor_name")}</span>
            <input value={createNodeDraft.label || ""} onChange={(event) => onCreateNodeDraftChange({ ...createNodeDraft, label: event.target.value, name: event.target.value })} />
          </label>
          <label className="field-stack compact">
            <span>{t("editor_type")}</span>
            <input value={createNodeDraft.type || "CONCEPT"} onChange={(event) => onCreateNodeDraftChange({ ...createNodeDraft, type: event.target.value })} />
          </label>
          <label className="field-stack">
            <span>{t("editor_summary")}</span>
            <textarea value={createNodeDraft.short_gloss || ""} onChange={(event) => onCreateNodeDraftChange({ ...createNodeDraft, short_gloss: event.target.value, description: event.target.value })} />
          </label>
          <label className="field-stack">
            <span>{t("editor_what_it_is")}</span>
            <textarea value={createNodeDraft.what_it_is || ""} onChange={(event) => onCreateNodeDraftChange({ ...createNodeDraft, what_it_is: event.target.value, plain_explanation: event.target.value })} />
          </label>
        </div>
      </section>

      <section className="controller-card wide">
        <div className="panel-heading compact">
          <div>
            <h3>{t("editor_create_relation_title")}</h3>
            <p>{t("editor_create_relation_text")}</p>
          </div>
          <button type="button" className="button-secondary" onClick={onCreateRelation} disabled={creatingRelation || !relationDraft.src_id || !relationDraft.dst_id || !relationDraft.type}>
            {creatingRelation ? t("editor_saving") : t("editor_create_relation")}
          </button>
        </div>
        <div className="panel-form grid-two">
          <label className="field-stack compact">
            <span>{t("editor_source")}</span>
            <input value={relationDraft.src_id || selectedNodeId || ""} onChange={(event) => onRelationDraftChange({ ...relationDraft, src_id: event.target.value })} />
          </label>
          <label className="field-stack compact">
            <span>{t("editor_target")}</span>
            <input
              list="visible-node-targets"
              value={relationDraft.dst_id || ""}
              onChange={(event) => onRelationDraftChange({ ...relationDraft, dst_id: event.target.value })}
            />
            <datalist id="visible-node-targets">
              {visibleNodes.map((node) => <option key={node.id} value={node.id}>{node.label || node.name || node.id}</option>)}
            </datalist>
          </label>
          <label className="field-stack compact span-2">
            <span>{t("editor_relation_type")}</span>
            <select value={relationDraft.type || "RELATED_TO"} onChange={(event) => onRelationDraftChange({ ...relationDraft, type: event.target.value })}>
              {EDGE_TYPE_OPTIONS.map((item) => <option key={item} value={item}>{item}</option>)}
            </select>
          </label>
          <label className="field-stack compact">
            <span>{t("inspector_weight")}</span>
            <input type="number" min="0" step="0.05" value={relationDraft.weight ?? 1} onChange={(event) => onRelationDraftChange({ ...relationDraft, weight: Number(event.target.value || 0) })} />
          </label>
          <label className="field-stack compact">
            <span>{t("inspector_confidence")}</span>
            <input type="number" min="0" max="1" step="0.05" value={relationDraft.confidence ?? 0.7} onChange={(event) => onRelationDraftChange({ ...relationDraft, confidence: Number(event.target.value || 0) })} />
          </label>
        </div>
      </section>
    </section>
  );
}

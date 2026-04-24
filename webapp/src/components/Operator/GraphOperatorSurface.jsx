import React from 'react';

import { GraphWorkspace } from '../Graph/GraphWorkspace';
import {
  nodeCanonicalEnglishExplanation,
  nodeDescription,
  nodeIdentity,
  nodeLocalizedExplanation,
  nodeReflectionForm,
  nodeViewDescription,
} from '../../lib/operatorFormatters';

export function GraphOperatorSurface({
  graphData,
  loops,
  selection,
  setSelection,
  branchPayloads,
  buildLoopSets,
  closeBranchWindow,
  openBranchWindow,
  nodeTitleById,
  graphQuery,
  onGraphQueryChange,
  onGraphSearch,
  onGraphRebuild,
  rebuildingGraph,
  onCreateGraphNode,
  onDisconnectEdge,
  onConnectNodes,
  onMergeNodes,
  onDeleteSelection,
  onQuarantineNode,
  graphReflectionMode,
  onToggleReflectionMode,
  onGraphRethinkPreview,
  onGraphRethink,
  graphPreviewing,
  graphReflecting,
  graphDraft,
  setGraphDraft,
  pendingActionLabel,
  graphSubgraphLimit,
  onGraphSubgraphLimitChange,
  graphLayoutSpread,
  onGraphLayoutSpreadChange,
  showEdgeLabels,
  onToggleEdgeLabels,
  graphRootNodeId,
  onFocusSelection,
  onResetGraphFocus,
  nodeView,
  nodeViewLoading,
  selectedNodeRelations,
  graphRethinkPreview,
  selectedPreview,
  t,
}) {
  const selectedNode = selection?.kind === 'node' ? selection.payload : null;
  const selectedReflectionForm = nodeReflectionForm(nodeView);
  const selectedCanonicalEnglishExplanation = nodeCanonicalEnglishExplanation(nodeView);
  const selectedLocalizedExplanation = nodeLocalizedExplanation(nodeView);
  const previewResults = graphRethinkPreview?.results || [];

  return (
    <div className="graph-workspace-shell operator-surface-grid-single">
      <div className="graph-stage">
        <div className="graph-toolbar glass-panel graph-toolbar-extended">
          <label className="field-stack compact grow">
            <span>{t('graph_query')}</span>
            <input value={graphQuery} onChange={(event) => onGraphQueryChange(event.target.value)} placeholder={t('graph_query_placeholder')} />
          </label>
          <label className="field-stack compact graph-inline-field">
            <span>{t('graph_limit')}</span>
            <input type="number" min="4" max="48" value={graphSubgraphLimit} onChange={(event) => onGraphSubgraphLimitChange(event.target.value)} />
          </label>
          <label className="field-stack compact graph-inline-field">
            <span>{t('graph_spread')}</span>
            <input type="range" min="0.9" max="1.9" step="0.05" value={graphLayoutSpread} onChange={(event) => onGraphLayoutSpreadChange(event.target.value)} />
          </label>
          <label className="checkbox-row compact-inline">
            <input type="checkbox" checked={showEdgeLabels} onChange={(event) => onToggleEdgeLabels(event.target.checked)} />
            <span>{t('graph_edge_labels')}</span>
          </label>
          <button type="button" onClick={onGraphSearch}>{t('graph_search')}</button>
          <button type="button" onClick={onGraphRebuild} disabled={rebuildingGraph}>
            {rebuildingGraph ? t('graph_rebuilding') : t('graph_rebuild')}
          </button>
          <button type="button" onClick={onCreateGraphNode}>{t('graph_add_node')}</button>
          <button type="button" onClick={onDisconnectEdge} disabled={selection?.kind !== 'edge'}>{t('graph_disconnect')}</button>
          <button type="button" onClick={onConnectNodes} disabled={selection?.kind !== 'node'}>{t('graph_connect')}</button>
          <button type="button" onClick={onMergeNodes} disabled={selection?.kind !== 'node'}>{t('graph_merge')}</button>
          <button type="button" onClick={onDeleteSelection} disabled={!selection}>{t('graph_delete')}</button>
          <button type="button" className="button-secondary" onClick={onQuarantineNode} disabled={selection?.kind !== 'node'}>{t('graph_quarantine')}</button>
          <button type="button" className="button-secondary" onClick={onFocusSelection} disabled={selection?.kind !== 'node'}>{t('graph_focus_selected')}</button>
          <button type="button" className="button-secondary" onClick={onResetGraphFocus} disabled={!graphRootNodeId}>{t('graph_reset_focus')}</button>
        </div>

        <div className="graph-rethink-toolbar glass-panel">
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={graphReflectionMode}
              onChange={(event) => onToggleReflectionMode(event.target.checked)}
            />
            <span>{t('graph_rethink_mode')}</span>
          </label>
          {graphReflectionMode && (
            <span className="graph-rethink-active-warning">{t('graph_rethink_active_mode_warning')}</span>
          )}
          <div className="graph-rethink-actions">
            <button type="button" onClick={onGraphRethinkPreview} disabled={graphPreviewing || graphReflecting}>
              {graphPreviewing ? t('graph_rethink_previewing') : t('graph_rethink_preview')}
            </button>
            <button
              type="button"
              onClick={onGraphRethink}
              disabled={graphReflecting || !graphRethinkPreview}
              title={!graphRethinkPreview ? t('graph_rethink_needs_preview') : undefined}
            >
              {graphReflecting ? t('graph_rethinking') : t('graph_rethink_apply')}
            </button>
          </div>
          {!graphRethinkPreview && !graphPreviewing && (
            <span className="graph-rethink-hint-inline">{t('graph_rethink_needs_preview')}</span>
          )}
        </div>

        <div className="graph-editor-panel glass-panel">
          <label className="field-stack compact">
            <span>{t('graph_new_node_name')}</span>
            <input value={graphDraft.name} onChange={(event) => setGraphDraft((current) => ({ ...current, name: event.target.value }))} placeholder="Human" />
          </label>
          <label className="field-stack compact">
            <span>{t('graph_new_node_type')}</span>
            <select value={graphDraft.node_type} onChange={(event) => setGraphDraft((current) => ({ ...current, node_type: event.target.value }))}>
              <option value="CONCEPT">CONCEPT</option>
              <option value="PERSON">PERSON</option>
              <option value="PHENOMENON">PHENOMENON</option>
              <option value="OBJECT">OBJECT</option>
            </select>
          </label>
          <label className="field-stack compact">
            <span>{t('graph_translation_line')}</span>
            <input
              value={graphDraft.translation_line}
              onChange={(event) => setGraphDraft((current) => ({ ...current, translation_line: event.target.value }))}
              placeholder="Human: человек, մարդ"
            />
          </label>
          <label className="field-stack compact grow span-2">
            <span>{t('graph_new_node_description')}</span>
            <input
              value={graphDraft.description}
              onChange={(event) => setGraphDraft((current) => ({ ...current, description: event.target.value }))}
              placeholder={t('graph_new_node_description_placeholder')}
            />
          </label>
        </div>

        <div className="graph-inline-status">
          {pendingActionLabel || (graphReflectionMode ? t('graph_rethink_hint') : t('graph_branch_hint'))}
          {graphData.diagnostics ? (
            <span className="graph-inline-metrics">
              {` | duplicate=${graphData.diagnostics.duplicate_rate ?? 0} orphan=${graphData.diagnostics.orphan_rate ?? 0} suspect=${graphData.diagnostics.suspect_node_count ?? 0} archived=${graphData.diagnostics.archived_node_count ?? 0}`}
            </span>
          ) : null}
        </div>

        <GraphWorkspace
          nodes={graphData.nodes || []}
          edges={graphData.edges || []}
          diagnostics={graphData.diagnostics}
          loops={loops}
          selectedNodeId={selection?.kind === 'node' ? selection.id : ''}
          selectedEdgeKey={selection?.kind === 'edge' ? (selection.payload?.id || selection.id) : ''}
          highlightedNodeIds={new Set(graphData.seed_node_ids || [])}
          highlightedEdgeKeys={new Set()}
          searchHitIds={new Set(graphData.seed_node_ids || [])}
          onSelectNode={(node) => setSelection({ kind: 'node', id: node.id, payload: node })}
          onSelectEdge={(edge) => setSelection({ kind: 'edge', id: edge.id || `${edge.from || edge.src_id}|${edge.type}|${edge.to || edge.dst_id}`, payload: edge })}
          onClearSelection={() => setSelection(null)}
          onOpenBranch={(node, parentRootId) => openBranchWindow(node, parentRootId || graphRootNodeId || graphData.seed_node_ids?.[0] || '')}
          rootNodeId={graphRootNodeId || graphData.seed_node_ids?.[0] || ''}
          windowTitle={t('graph_workspace')}
          windowSubtitle={graphData.query ? `${t('graph_query')}: ${graphData.query}` : ''}
          layoutSpread={graphLayoutSpread}
          showEdgeLabels={showEdgeLabels}
          t={t}
        />

        {branchPayloads.length ? (
          <div className="graph-branch-grid">
            {branchPayloads.map((branch) => (
              <GraphWorkspace
                key={branch.id}
                nodes={branch.payload.nodes}
                edges={branch.payload.edges}
                loops={buildLoopSets(branch.payload.edges)}
                selectedNodeId={selection?.kind === 'node' ? selection.id : ''}
                selectedEdgeKey={selection?.kind === 'edge' ? (selection.payload?.id || selection.id) : ''}
                highlightedNodeIds={new Set([branch.rootNodeId])}
                highlightedEdgeKeys={branch.payload.highlightedEdgeKeys}
                searchHitIds={new Set([branch.rootNodeId])}
                onSelectNode={(node) => setSelection({ kind: 'node', id: node.id, payload: node })}
                onSelectEdge={(edge) => setSelection({ kind: 'edge', id: edge.id || `${edge.from || edge.src_id}|${edge.type}|${edge.to || edge.dst_id}`, payload: edge })}
                onClearSelection={() => setSelection(null)}
                onOpenBranch={(node) => openBranchWindow(node, branch.rootNodeId)}
                rootNodeId={branch.rootNodeId}
                windowTitle={`${t('graph_branch_window')}: ${nodeTitleById(branch.rootNodeId)}`}
                windowSubtitle={branch.subtitle}
                onCloseWindow={() => closeBranchWindow(branch.id)}
                isBranchWindow
                layoutSpread={graphLayoutSpread * 0.92}
                showEdgeLabels={showEdgeLabels}
                t={t}
              />
            ))}
          </div>
        ) : null}
      </div>

      <section className="workspace-panel glass-panel node-answer-panel node-editor-shell">
        <header className="panel-heading compact">
          <div>
            <p className="eyebrow">{t('graph_operator_inspector')}</p>
            <h2>{selectedNode ? (selectedNode.name || selectedNode.id) : t('graph_no_selection')}</h2>
          </div>
        </header>
        {selectedNode ? (
          <div className="node-answer-grid">
            <section>
              <h3>{t('graph_node_identity')}</h3>
              <p>{nodeIdentity(selectedNode)}</p>
              <div className="operator-kv-grid">
                <div><span>{t('graph_node_state')}</span><strong>{selectedNode.lifecycle_state || 'active'}</strong></div>
                <div><span>{t('graph_node_importance')}</span><strong>{Number(selectedNode.importance ?? 0).toFixed(2)}</strong></div>
                <div><span>{t('graph_node_confidence')}</span><strong>{Number(selectedNode.confidence ?? 0).toFixed(2)}</strong></div>
                <div><span>{t('graph_node_frequency')}</span><strong>{Number(selectedNode.frequency ?? 0).toFixed(0)}</strong></div>
              </div>
              {selectedNode?.cluster_label ? <p>{`cluster: ${selectedNode.cluster_label}`}</p> : null}
            </section>
            <section>
              <h3>{t('graph_node_description')}</h3>
              {nodeViewLoading ? (
                <p>{t('graph_node_loading')}</p>
              ) : (
                <div className="node-description-stack">
                  <p>{nodeView ? nodeViewDescription(nodeView) : nodeDescription(selectedNode)}</p>
                  {selectedCanonicalEnglishExplanation ? (
                    <div className="node-language-block">
                      <strong>{t('graph_canonical_english')}</strong>
                      <p>{selectedCanonicalEnglishExplanation}</p>
                    </div>
                  ) : null}
                  {selectedLocalizedExplanation ? (
                    <div className="node-language-block translated">
                      <strong>{t('graph_localized_explanation')}</strong>
                      <p>{selectedLocalizedExplanation}</p>
                    </div>
                  ) : null}
                </div>
              )}
            </section>
            {selectedReflectionForm ? (
              <section>
                <h3>{t('graph_node_rethinking')}</h3>
                <ul className="dense-list">
                  {selectedReflectionForm.who_or_what ? <li><strong>{t('graph_form_who')}</strong> {selectedReflectionForm.who_or_what}</li> : null}
                  {selectedReflectionForm.what_can_it_do ? <li><strong>{t('graph_form_can')}</strong> {selectedReflectionForm.what_can_it_do}</li> : null}
                  {selectedReflectionForm.how_does_it_work ? <li><strong>{t('graph_form_how')}</strong> {selectedReflectionForm.how_does_it_work}</li> : null}
                  {selectedReflectionForm.why_it_matters ? <li><strong>{t('graph_form_why')}</strong> {selectedReflectionForm.why_it_matters}</li> : null}
                  {Array.isArray(selectedReflectionForm.suggested_links) && selectedReflectionForm.suggested_links.length ? (
                    <li><strong>{t('graph_form_links')}</strong> {selectedReflectionForm.suggested_links.join(' | ')}</li>
                  ) : null}
                </ul>
              </section>
            ) : null}
            {selectedPreview ? (
              <section className="rethink-preview-panel">
                <h3>{t('graph_rethink_preview_title')}</h3>
                <ul className="dense-list">
                  {selectedPreview.preview?.description ? <li><strong>{t('graph_rethink_will_update')}</strong> {selectedPreview.preview.description}</li> : null}
                  {Array.isArray(selectedPreview.preview?.links) && selectedPreview.preview.links.length ? (
                    <li>
                      <strong>{t('graph_rethink_will_link')}</strong>
                      {' '}
                      {selectedPreview.preview.links.map((item) => `${item.relation_type} ${item.name}`).join(' | ')}
                    </li>
                  ) : null}
                  {Array.isArray(selectedPreview.skippedSuggestions) && selectedPreview.skippedSuggestions.length ? (
                    <li>
                      <strong>{t('graph_rethink_skipped')}</strong>
                      {' '}
                      {selectedPreview.skippedSuggestions.map((item) => `${item.name || 'unknown'}:${item.reason || 'skipped'}`).join(' | ')}
                    </li>
                  ) : null}
                </ul>
              </section>
            ) : null}
            {!selectedPreview && graphRethinkPreview?.previewOnly && previewResults.length ? (
              <section className="rethink-preview-panel">
                <h3>{t('graph_rethink_preview_title')}</h3>
                <ul className="dense-list">
                  {previewResults.map((item) => (
                    <li key={item.nodeId || item.nodeName}>
                      <strong>{item.nodeName || item.nodeId}</strong>
                      {' '}
                      {item.preview?.links?.length
                        ? item.preview.links.map((link) => `${link.relation_type} ${link.name}`).join(' | ')
                        : t('graph_rethink_no_preview')}
                    </li>
                  ))}
                </ul>
              </section>
            ) : null}
            <section>
              <h3>{t('graph_node_relations')}</h3>
              {selectedNodeRelations.length ? (
                <ul className="dense-list">
                  {selectedNodeRelations.map((item) => (
                    <li key={item.key}>
                      <strong>{item.direction === 'out' ? '->' : '<-'}</strong> {item.type} {item.other} <span>({item.weight})</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p>{t('graph_no_selection')}</p>
              )}
            </section>
          </div>
        ) : (
          <p>{t('graph_no_selection')}</p>
        )}
      </section>
    </div>
  );
}

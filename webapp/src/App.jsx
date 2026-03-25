import React, { useEffect, useMemo, useState } from 'react';
import {
  connectCognitiveGraphNodes,
  createCognitiveSession,
  createCognitiveGraphNode,
  deleteCognitiveGraphEdge,
  deleteCognitiveGraphNode,
  getCognitiveDebugMetrics,
  getCognitiveDebugTraces,
  getCognitiveGraph,
  getCognitiveGraphHealth,
  getCognitiveGraphNodeView,
  getCognitivePersonality,
  getCognitiveGraphSubgraph,
  getCognitiveSession,
  getHealth,
  listCognitivePersonalities,
  listCognitiveSessions,
  mergeCognitiveGraphNodes,
  rebuildCognitiveGraph,
  reviewCognitiveGraphNode,
  rethinkCognitiveGraphNodes,
  respondCognitiveChat,
  uploadCognitiveFiles,
} from './api';
import { Sidebar } from './components/Layout/Sidebar';
import { TopBar } from './components/Layout/TopBar';
import { ChatSurface } from './components/Operator/ChatSurface';
import { DiagnosticsSurface } from './components/Operator/DiagnosticsSurface';
import { FilesIngestionSurface } from './components/Operator/FilesIngestionSurface';
import { GraphOperatorSurface } from './components/Operator/GraphOperatorSurface';
import { PersonaInspectionSurface } from './components/Operator/PersonaInspectionSurface';
import { extractNeighborhoodPayload, shortestPathBetween } from './lib/graphView';
import { createTranslator, LANGUAGE_OPTIONS } from './lib/i18n';
import { normalizeRethinkPreview } from './lib/operatorFormatters';

const LANGUAGE_STORAGE_KEY = 'workspace_ui_language_mvp';

function buildLoopSets(edges) {
  const pairCounts = new Set();
  const twoCycleKeys = new Set();
  const loopNodeIds = new Set();
  for (const edge of edges || []) {
    const src = String(edge.from || edge.src_id || '');
    const dst = String(edge.to || edge.dst_id || '');
    if (!src || !dst) continue;
    if (src === dst) {
      loopNodeIds.add(src);
      continue;
    }
    pairCounts.add(`${src}|${dst}`);
  }
  for (const edge of edges || []) {
    const src = String(edge.from || edge.src_id || '');
    const dst = String(edge.to || edge.dst_id || '');
    if (!src || !dst || src === dst) continue;
    if (pairCounts.has(`${dst}|${src}`)) {
      twoCycleKeys.add(`${src}|${dst}`);
      twoCycleKeys.add(`${dst}|${src}`);
      loopNodeIds.add(src);
      loopNodeIds.add(dst);
    }
  }
  return { loopNodeIds, twoCycleKeys };
}

function nodeRelations(nodeId, edges, nodesById) {
  if (!nodeId) return [];
  return (edges || [])
    .filter((edge) => String(edge.from || edge.src_id || '') === String(nodeId) || String(edge.to || edge.dst_id || '') === String(nodeId))
    .map((edge) => {
      const from = String(edge.from || edge.src_id || '');
      const to = String(edge.to || edge.dst_id || '');
      const otherId = from === String(nodeId) ? to : from;
      const otherNode = nodesById.get(otherId);
      return {
        key: `${from}|${edge.type}|${to}`,
        type: edge.type,
        weight: Number(edge.weight ?? 1).toFixed(2),
        other: otherNode?.name || otherNode?.id || otherId,
        direction: from === String(nodeId) ? 'out' : 'in',
      };
    });
}

export default function App() {
  const [uiLanguage, setUiLanguage] = useState(() => {
    if (typeof window === 'undefined') return 'ru';
    return window.localStorage.getItem(LANGUAGE_STORAGE_KEY) || 'ru';
  });
  const t = useMemo(() => createTranslator(uiLanguage), [uiLanguage]);

  const [workspace, setWorkspace] = useState('chat');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [refreshing, setRefreshing] = useState(false);
  const [health, setHealth] = useState(null);

  const [sessions, setSessions] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState('');
  const [activeSession, setActiveSession] = useState(null);
  const [chatInput, setChatInput] = useState('');
  const [chatRunning, setChatRunning] = useState(false);
  const [chatProgress, setChatProgress] = useState('');
  const [lastChatResult, setLastChatResult] = useState(null);
  const [activeTrace, setActiveTrace] = useState(null);
  const [diagnostics, setDiagnostics] = useState({ metrics: null, traces: [], graphHealth: null });
  const [diagnosticsLoading, setDiagnosticsLoading] = useState(false);

  const [personalities, setPersonalities] = useState([]);
  const [selectedPersonality, setSelectedPersonality] = useState('');
  const [personaDetail, setPersonaDetail] = useState(null);
  const [personaDetailLoading, setPersonaDetailLoading] = useState(false);
  const [uploadingFiles, setUploadingFiles] = useState(false);
  const [lastUploadResult, setLastUploadResult] = useState(null);

  const [graphData, setGraphData] = useState({ nodes: [], edges: [], seed_node_ids: [], query: '', diagnostics: null, clusters: [] });
  const [graphQuery, setGraphQuery] = useState('');
  const [graphSubgraphLimit, setGraphSubgraphLimit] = useState(16);
  const [graphLayoutSpread, setGraphLayoutSpread] = useState(1.34);
  const [showEdgeLabels, setShowEdgeLabels] = useState(true);
  const [graphRootNodeId, setGraphRootNodeId] = useState('');
  const [rebuildingGraph, setRebuildingGraph] = useState(false);
  const [graphContentLanguage, setGraphContentLanguage] = useState('');
  const [graphReflectionMode, setGraphReflectionMode] = useState(false);
  const [graphReflecting, setGraphReflecting] = useState(false);
  const [graphPreviewing, setGraphPreviewing] = useState(false);
  const [graphRethinkPreview, setGraphRethinkPreview] = useState(null);
  const [selection, setSelection] = useState(null);
  const [nodeView, setNodeView] = useState(null);
  const [nodeViewLoading, setNodeViewLoading] = useState(false);
  const [branchWindows, setBranchWindows] = useState([]);
  const [pendingGraphAction, setPendingGraphAction] = useState(null);
  const [graphDraft, setGraphDraft] = useState({
    name: '',
    node_type: 'CONCEPT',
    translation_line: '',
    description: '',
  });
  const graphTextLanguage = graphContentLanguage || uiLanguage;

  useEffect(() => {
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(LANGUAGE_STORAGE_KEY, uiLanguage);
    }
  }, [uiLanguage]);

  useEffect(() => {
    const visibleIds = new Set((graphData.nodes || []).map((node) => String(node.id)));
    setBranchWindows((current) => current.filter((item) => visibleIds.has(String(item.rootNodeId))));
    if (pendingGraphAction && !visibleIds.has(String(pendingGraphAction.anchorNodeId || ''))) {
      setPendingGraphAction(null);
    }
    if (graphRootNodeId && !visibleIds.has(String(graphRootNodeId))) {
      setGraphRootNodeId('');
    }
  }, [graphData.nodes, pendingGraphAction, graphRootNodeId]);

  useEffect(() => {
    setGraphRethinkPreview(null);
  }, [selection?.id, selection?.kind, graphReflectionMode, graphQuery, graphData.query]);

  useEffect(() => {
    if (workspace === 'persona') {
      const target = selectedPersonality || personalities[0]?.name || '';
      if (target) {
        void loadPersonaDetail(target).catch((detailError) => setError(detailError.message || String(detailError)));
      }
    }
  }, [workspace, selectedPersonality, personalities]);

  useEffect(() => {
    if (workspace === 'diagnostics') {
      void loadDiagnostics(activeSessionId, activeTrace?.request_id || '').catch((diagError) => setError(diagError.message || String(diagError)));
    }
  }, [workspace, activeSessionId]);

  function applyGraphPayload(result, query = '') {
    setGraphData({
      nodes: result.nodes || [],
      edges: result.edges || [],
      seed_node_ids: result.seed_node_ids || [],
      query: result.query || query || '',
      diagnostics: result.diagnostics || null,
      clusters: result.clusters || result.diagnostics?.clusters || [],
    });
  }

  function nodeTitleById(nodeId) {
    return (graphData.nodes || []).find((node) => String(node.id) === String(nodeId))?.name || nodeId;
  }

  function openBranchWindow(node, parentRootId = '') {
    if (!node?.id) return;
    const rootNodeId = String(node.id);
    setBranchWindows((current) => {
      const existing = current.find((item) => String(item.rootNodeId) === rootNodeId);
      if (existing) {
        return [existing, ...current.filter((item) => item.id !== existing.id)];
      }
      return [
        {
          id: `branch-${rootNodeId}`,
          rootNodeId,
          parentRootId: String(parentRootId || ''),
        },
        ...current,
      ].slice(0, 6);
    });
  }

  function closeBranchWindow(branchId) {
    setBranchWindows((current) => current.filter((item) => item.id !== branchId));
  }

  async function handleCreateSession() {
    const result = await createCognitiveSession({ title: 'New session' });
    const session = result.session;
    setSessions((current) => [session, ...current.filter((item) => item.session_id !== session.session_id)]);
    setActiveSession(session);
    setActiveSessionId(session.session_id);
    return session;
  }

  async function loadSession(sessionId) {
    const session = await getCognitiveSession(sessionId);
    setActiveSession(session);
    setActiveSessionId(session.session_id);
    return session;
  }

  async function loadGraph(query, limit = graphSubgraphLimit) {
    const result = query ? await getCognitiveGraphSubgraph(query, limit) : await getCognitiveGraph();
    applyGraphPayload(result, query);
    return result;
  }

  async function refreshGraphView(preferredQuery = '') {
    const nextQuery = String(preferredQuery || graphQuery || graphData.query || '').trim();
    return loadGraph(nextQuery);
  }

  async function loadNodeView(nodeId) {
    if (!nodeId) {
      setNodeView(null);
      return null;
    }
    const result = await getCognitiveGraphNodeView(nodeId, graphTextLanguage);
    setNodeView(result);
    return result;
  }

  async function loadPersonaDetail(name) {
    const cleanName = String(name || '').trim();
    if (!cleanName) {
      setPersonaDetail(null);
      return null;
    }
    setPersonaDetailLoading(true);
    try {
      const result = await getCognitivePersonality(cleanName);
      setPersonaDetail(result);
      return result;
    } finally {
      setPersonaDetailLoading(false);
    }
  }

  async function loadDiagnostics(sessionId = activeSessionId, traceId = '') {
    setDiagnosticsLoading(true);
    try {
      const [metricsResult, tracesResult, graphHealthResult] = await Promise.all([
        getCognitiveDebugMetrics(),
        getCognitiveDebugTraces({ sessionId, limit: 20 }),
        getCognitiveGraphHealth(),
      ]);
      const traceRows = tracesResult.traces || [];
      const chosenTrace = traceId
        ? traceRows.find((item) => String(item.request_id) === String(traceId))
        : traceRows[0] || null;
      setDiagnostics({
        metrics: metricsResult.metrics || null,
        traces: traceRows,
        graphHealth: graphHealthResult.graph_health || null,
      });
      setActiveTrace(chosenTrace || null);
      return { metrics: metricsResult.metrics, traces: traceRows, graphHealth: graphHealthResult.graph_health };
    } finally {
      setDiagnosticsLoading(false);
    }
  }

  async function handleCreateGraphNode() {
    if (!graphDraft.name.trim()) return;
    setError('');
    try {
      await createCognitiveGraphNode({
        name: graphDraft.name.trim(),
        node_type: graphDraft.node_type,
        translation_line: graphDraft.translation_line.trim(),
        description: graphDraft.description.trim(),
      });
      await refreshGraphView(graphDraft.name.trim());
      setGraphDraft({ name: '', node_type: 'CONCEPT', translation_line: '', description: '' });
      setWorkspace('graph');
    } catch (graphError) {
      setError(graphError.message || String(graphError));
    }
  }

  async function handleDeleteSelection() {
    if (!selection) return;
    setError('');
    try {
      if (selection.kind === 'edge') {
        const edgeId = selection.payload?.id || selection.payload?.edge_id || '';
        if (!edgeId) throw new Error('Select an edge with a persistent id first.');
        await deleteCognitiveGraphEdge(edgeId);
      } else if (selection.kind === 'node') {
        await deleteCognitiveGraphNode(selection.payload?.id || selection.id);
      } else {
        return;
      }
      setSelection(null);
      setPendingGraphAction(null);
      await refreshGraphView();
    } catch (graphError) {
      setError(graphError.message || String(graphError));
    }
  }

  async function handleQuarantineSelection() {
    if (selection?.kind !== 'node') {
      setError('Select a node first.');
      return;
    }
    setError('');
    try {
      await reviewCognitiveGraphNode(selection.id, {
        lifecycle_state: 'suspect',
        reason: 'operator_quarantine',
      });
      await refreshGraphView();
      await loadNodeView(selection.id);
    } catch (graphError) {
      setError(graphError.message || String(graphError));
    }
  }

  async function handleDisconnectEdge() {
    if (selection?.kind !== 'edge') {
      setError('Select an edge to remove it.');
      return;
    }
    await handleDeleteSelection();
  }

  async function handleConnectNodes() {
    if (selection?.kind !== 'node') {
      setError('Select a node first.');
      return;
    }
    if (!pendingGraphAction || pendingGraphAction.type !== 'connect') {
      setPendingGraphAction({ type: 'connect', anchorNodeId: selection.id });
      return;
    }
    if (pendingGraphAction.anchorNodeId === selection.id) {
      setError('Select another node and press connect again.');
      return;
    }
    setError('');
    try {
      await connectCognitiveGraphNodes({
        from_id: pendingGraphAction.anchorNodeId,
        to_id: selection.id,
        relation_type: 'RELATED_TO',
      });
      setPendingGraphAction(null);
      await refreshGraphView();
    } catch (graphError) {
      setError(graphError.message || String(graphError));
    }
  }

  async function handleMergeNodes() {
    if (selection?.kind !== 'node') {
      setError('Select a node first.');
      return;
    }
    if (!pendingGraphAction || pendingGraphAction.type !== 'merge') {
      setPendingGraphAction({ type: 'merge', anchorNodeId: selection.id });
      return;
    }
    if (pendingGraphAction.anchorNodeId === selection.id) {
      setError('Select another node and press merge again.');
      return;
    }
    setError('');
    try {
      await mergeCognitiveGraphNodes({
        primary_id: pendingGraphAction.anchorNodeId,
        secondary_id: selection.id,
      });
      setSelection(null);
      setPendingGraphAction(null);
      await refreshGraphView();
    } catch (graphError) {
      setError(graphError.message || String(graphError));
    }
  }

  async function bootstrap() {
    setLoading(true);
    setError('');
    try {
      const [nextHealth, personalityResult, sessionsResult] = await Promise.all([
        getHealth(),
        listCognitivePersonalities(),
        listCognitiveSessions(),
      ]);
      setHealth(nextHealth);
      setPersonalities(personalityResult.personalities || []);
      const nextSessions = sessionsResult.sessions || [];
      setSessions(nextSessions);
      if (nextSessions.length) {
        await loadSession(nextSessions[0].session_id);
      } else {
        await handleCreateSession();
      }
      await loadGraph('');
    } catch (bootError) {
      setError(bootError.message || String(bootError));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void bootstrap();
  }, []);

  useEffect(() => {
    let cancelled = false;
    if (selection?.kind !== 'node' || !selection.id) {
      setNodeView(null);
      setNodeViewLoading(false);
      return () => {
        cancelled = true;
      };
    }
    setNodeViewLoading(true);
    void getCognitiveGraphNodeView(selection.id, graphTextLanguage)
      .then((result) => {
        if (!cancelled) {
          setNodeView(result);
        }
      })
      .catch((viewError) => {
        if (!cancelled) {
          setError(viewError.message || String(viewError));
          setNodeView(null);
        }
      })
      .finally(() => {
        if (!cancelled) {
          setNodeViewLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [selection?.id, selection?.kind, graphTextLanguage]);

  async function refreshAll() {
    setRefreshing(true);
    try {
      await bootstrap();
    } finally {
      setRefreshing(false);
    }
  }

  async function handleRunChat() {
    const message = chatInput.trim();
    if (!message || chatRunning) return;
    setChatRunning(true);
    setChatProgress(t('chat_running'));
    setError('');
    try {
      const sessionId = activeSessionId || (await handleCreateSession()).session_id;
      const result = await respondCognitiveChat({
        message,
        session_id: sessionId,
        language: uiLanguage,
        personality_name: selectedPersonality,
      });
      setLastChatResult(result);
      setGraphContentLanguage(result?.analysis?.user_state?.language || uiLanguage);
      setChatInput('');
      if (result.session) {
        setActiveSession(result.session);
        setActiveSessionId(result.session.session_id);
        setSessions((current) => [result.session, ...current.filter((item) => item.session_id !== result.session.session_id)]);
      } else {
        await loadSession(sessionId);
      }
      const followupQuery = result.current_entity || selectedPersonality || message;
      await loadGraph(followupQuery);
      if (result.trace_id) {
        await loadDiagnostics(sessionId, result.trace_id);
      }
      if (result.persona_name) {
        const personaDisplayName = result.persona_selection?.persona_name || result.persona_name;
        setSelectedPersonality(personaDisplayName);
        await loadPersonaDetail(personaDisplayName);
      }
      setWorkspace('chat');
    } catch (runError) {
      setError(runError.message || String(runError));
    } finally {
      setChatRunning(false);
      setChatProgress('');
    }
  }

  async function handleUploadFiles(fileList) {
    const files = Array.from(fileList || []);
    if (!files.length || uploadingFiles) return;
    setUploadingFiles(true);
    setError('');
    try {
      const sessionId = activeSessionId || (await handleCreateSession()).session_id;
      const uploadResult = await uploadCognitiveFiles(sessionId, files);
      setLastUploadResult(uploadResult);
      await loadSession(sessionId);
      await loadGraph(graphQuery.trim() || selectedPersonality || activeSession?.title || files[0].name || '');
      setWorkspace('files');
    } catch (uploadError) {
      setError(uploadError.message || String(uploadError));
    } finally {
      setUploadingFiles(false);
    }
  }

  async function handleGraphSearch() {
    try {
      await loadGraph(graphQuery.trim());
      setWorkspace('graph');
    } catch (graphError) {
      setError(graphError.message || String(graphError));
    }
  }

  async function handleGraphRebuild() {
    setRebuildingGraph(true);
    setError('');
    try {
      await rebuildCognitiveGraph({ session_id: activeSessionId, personality_name: selectedPersonality });
      await loadGraph(graphQuery.trim() || selectedPersonality || activeSession?.title || '');
    } catch (rebuildError) {
      setError(rebuildError.message || String(rebuildError));
    } finally {
      setRebuildingGraph(false);
    }
  }

  function handleFocusSelection() {
    if (selection?.kind !== 'node') {
      setError('Select a node first.');
      return;
    }
    setGraphRootNodeId(String(selection.id));
  }

  function handleResetGraphFocus() {
    setGraphRootNodeId('');
  }

  async function handleGraphRethink() {
    const visibleNodeIds = (graphData.nodes || []).map((node) => String(node.id || '')).filter(Boolean);
    const selectedNodeId = selection?.kind === 'node' ? String(selection.id || '') : '';
    const nodeIds = graphReflectionMode
      ? visibleNodeIds.slice(0, 6)
      : (selectedNodeId ? [selectedNodeId] : visibleNodeIds.slice(0, 1));
    if (!nodeIds.length) {
      setError('Select a node or load a graph segment first.');
      return;
    }
    setGraphReflecting(true);
    setError('');
    try {
      const result = await rethinkCognitiveGraphNodes({
        node_ids: nodeIds,
        query: graphQuery.trim() || graphData.query || nodeTitleById(nodeIds[0]),
        limit: graphReflectionMode ? 6 : 1,
        context_budget: 4000,
        active_mode: graphReflectionMode,
        language: graphTextLanguage,
        preview_only: false,
      });
      setGraphRethinkPreview(normalizeRethinkPreview(result));
      await refreshGraphView(graphQuery.trim() || graphData.query || nodeTitleById(nodeIds[0]));
      if (selectedNodeId) {
        await loadNodeView(selectedNodeId);
      }
    } catch (graphError) {
      setError(graphError.message || String(graphError));
    } finally {
      setGraphReflecting(false);
    }
  }

  async function handleGraphRethinkPreview() {
    const visibleNodeIds = (graphData.nodes || []).map((node) => String(node.id || '')).filter(Boolean);
    const selectedNodeId = selection?.kind === 'node' ? String(selection.id || '') : '';
    const nodeIds = graphReflectionMode
      ? visibleNodeIds.slice(0, 6)
      : (selectedNodeId ? [selectedNodeId] : visibleNodeIds.slice(0, 1));
    if (!nodeIds.length) {
      setError('Select a node or load a graph segment first.');
      return;
    }
    setGraphPreviewing(true);
    setError('');
    try {
      const result = await rethinkCognitiveGraphNodes({
        node_ids: nodeIds,
        query: graphQuery.trim() || graphData.query || nodeTitleById(nodeIds[0]),
        limit: graphReflectionMode ? 6 : 1,
        context_budget: 4000,
        active_mode: graphReflectionMode,
        language: graphTextLanguage,
        preview_only: true,
      });
      setGraphRethinkPreview(normalizeRethinkPreview(result));
    } catch (graphError) {
      setError(graphError.message || String(graphError));
    } finally {
      setGraphPreviewing(false);
    }
  }

  const loops = useMemo(() => buildLoopSets(graphData.edges || []), [graphData.edges]);
  const nodesById = useMemo(() => new Map((graphData.nodes || []).map((node) => [String(node.id), node])), [graphData.nodes]);
  const selectedNode = selection?.kind === 'node' ? selection.payload : null;
  const selectedNodeRelations = useMemo(() => {
    if (nodeView?.how_it_acts && selectedNode?.id) {
      return (nodeView.how_it_acts || []).map((edge) => {
        const from = String(edge.from || edge.src_id || '');
        const to = String(edge.to || edge.dst_id || '');
        const otherId = from === String(selectedNode.id) ? to : from;
        const otherNode = nodesById.get(otherId);
        return {
          key: String(edge.id || `${from}|${edge.type}|${to}`),
          type: edge.type,
          weight: Number(edge.weight ?? 1).toFixed(2),
          other: otherNode?.name || (from === String(selectedNode.id) ? edge.to_name : edge.from_name) || otherId,
          direction: from === String(selectedNode.id) ? 'out' : 'in',
        };
      });
    }
    return nodeRelations(selectedNode?.id, graphData.edges || [], nodesById);
  }, [selectedNode, graphData.edges, nodeView, nodesById]);
  const branchPayloads = useMemo(
    () => branchWindows.map((branch) => {
      const payload = extractNeighborhoodPayload(branch.rootNodeId, graphData.nodes || [], graphData.edges || [], 2);
      const pathIds = branch.parentRootId ? shortestPathBetween(branch.parentRootId, branch.rootNodeId, graphData.edges || []) : [];
      const pathNames = pathIds.map((nodeId) => nodeTitleById(nodeId)).filter(Boolean);
      return {
        ...branch,
        payload,
        subtitle: pathNames.length ? pathNames.join(' -> ') : nodeTitleById(branch.rootNodeId),
      };
    }),
    [branchWindows, graphData.edges, graphData.nodes],
  );
  const pendingActionLabel = pendingGraphAction
    ? `${t(pendingGraphAction.type === 'connect' ? 'graph_pending_connect_from' : 'graph_pending_merge_from')} ${nodeTitleById(pendingGraphAction.anchorNodeId)}`
    : '';
  const previewResults = graphRethinkPreview?.results || [];
  const selectedPreview = selection?.kind === 'node'
    ? previewResults.find((item) => String(item.nodeId) === String(selection.id))
    : null;

  function renderWorkspaceSurface() {
    if (workspace === 'chat') {
      return (
        <ChatSurface
          session={activeSession}
          value={chatInput}
          onChange={setChatInput}
          running={chatRunning}
          progress={chatProgress}
          onRun={() => void handleRunChat()}
          lastChatResult={lastChatResult}
          activeTrace={activeTrace}
          t={t}
        />
      );
    }
    if (workspace === 'persona') {
      return (
        <PersonaInspectionSurface
          personalities={personalities}
          selectedPersonality={selectedPersonality}
          onSelectPersonality={(name) => {
            setSelectedPersonality(name);
            void loadPersonaDetail(name).catch((detailError) => setError(detailError.message || String(detailError)));
          }}
          personaDetail={personaDetail}
          loading={personaDetailLoading}
          lastChatResult={lastChatResult}
          t={t}
        />
      );
    }
    if (workspace === 'files') {
      return (
        <FilesIngestionSurface
          activeSessionId={activeSessionId}
          uploadingFiles={uploadingFiles}
          onUploadFiles={(files) => void handleUploadFiles(files)}
          lastUploadResult={lastUploadResult}
          t={t}
        />
      );
    }
    if (workspace === 'diagnostics') {
      return (
        <DiagnosticsSurface
          diagnostics={diagnostics}
          loading={diagnosticsLoading}
          onRefresh={() => void loadDiagnostics(activeSessionId, activeTrace?.request_id || '')}
          activeTrace={activeTrace}
          onSelectTrace={setActiveTrace}
          t={t}
        />
      );
    }
    return (
      <GraphOperatorSurface
        graphData={graphData}
        loops={loops}
        selection={selection}
        setSelection={setSelection}
        branchPayloads={branchPayloads}
        buildLoopSets={buildLoopSets}
        closeBranchWindow={closeBranchWindow}
        openBranchWindow={openBranchWindow}
        nodeTitleById={nodeTitleById}
        graphQuery={graphQuery}
        onGraphQueryChange={setGraphQuery}
        onGraphSearch={() => void handleGraphSearch()}
        onGraphRebuild={() => void handleGraphRebuild()}
        rebuildingGraph={rebuildingGraph}
        onCreateGraphNode={() => void handleCreateGraphNode()}
        onDisconnectEdge={() => void handleDisconnectEdge()}
        onConnectNodes={() => void handleConnectNodes()}
        onMergeNodes={() => void handleMergeNodes()}
        onDeleteSelection={() => void handleDeleteSelection()}
        onQuarantineNode={() => void handleQuarantineSelection()}
        graphReflectionMode={graphReflectionMode}
        onToggleReflectionMode={setGraphReflectionMode}
        onGraphRethinkPreview={() => void handleGraphRethinkPreview()}
        onGraphRethink={() => void handleGraphRethink()}
        graphPreviewing={graphPreviewing}
        graphReflecting={graphReflecting}
        graphDraft={graphDraft}
        setGraphDraft={setGraphDraft}
        pendingActionLabel={pendingActionLabel}
        graphSubgraphLimit={graphSubgraphLimit}
        onGraphSubgraphLimitChange={(value) => setGraphSubgraphLimit(Math.max(4, Math.min(48, Number(value) || 16)))}
        graphLayoutSpread={graphLayoutSpread}
        onGraphLayoutSpreadChange={(value) => setGraphLayoutSpread(Math.max(0.9, Math.min(1.9, Number(value) || 1.34)))}
        showEdgeLabels={showEdgeLabels}
        onToggleEdgeLabels={setShowEdgeLabels}
        graphRootNodeId={graphRootNodeId}
        onFocusSelection={handleFocusSelection}
        onResetGraphFocus={handleResetGraphFocus}
        nodeView={nodeView}
        nodeViewLoading={nodeViewLoading}
        selectedNodeRelations={selectedNodeRelations}
        graphRethinkPreview={graphRethinkPreview}
        selectedPreview={selectedPreview}
        t={t}
      />
    );
  }

  if (loading) {
    return <main className="app-shell loading-shell"><div className="empty-state large"><h2>Loading...</h2></div></main>;
  }

  return (
    <main className="app-shell">
      <TopBar
        health={health}
        language={uiLanguage}
        languageOptions={LANGUAGE_OPTIONS}
        onLanguageChange={setUiLanguage}
        onRefresh={refreshAll}
        refreshing={refreshing}
        t={t}
      />
      <div className="or-layout">
        <Sidebar
          workspace={workspace}
          onWorkspaceChange={setWorkspace}
          sessions={sessions}
          activeSessionId={activeSessionId}
          onSelectSession={(sessionId) => void loadSession(sessionId)}
          onCreateSession={() => void handleCreateSession()}
          personalities={personalities}
          selectedPersonality={selectedPersonality}
          onSelectPersonality={setSelectedPersonality}
          onUploadFiles={(files) => void handleUploadFiles(files)}
          uploadingFiles={uploadingFiles}
          graphQuery={graphQuery}
          onGraphQueryChange={setGraphQuery}
          onGraphSearch={() => void handleGraphSearch()}
          onGraphRebuild={() => void handleGraphRebuild()}
          rebuildingGraph={rebuildingGraph}
          t={t}
        />

        <section className="workspace-main">
          {error ? <div className="error-banner">{error}</div> : null}
          {renderWorkspaceSurface()}
        </section>
      </div>
    </main>
  );
}

import React, { useEffect, useMemo, useState } from 'react';
import {
  createCognitiveSession,
  getCognitiveGraph,
  getCognitiveGraphSubgraph,
  getCognitiveSession,
  getHealth,
  listCognitivePersonalities,
  listCognitiveSessions,
  rebuildCognitiveGraph,
  respondCognitiveChat,
  uploadCognitiveFiles,
} from './api';
import { ChatGraphPanel } from './components/Chat/ChatGraphPanel';
import { GraphWorkspace } from './components/Graph/GraphWorkspace';
import { Sidebar } from './components/Layout/Sidebar';
import { TopBar } from './components/Layout/TopBar';
import { createTranslator, LANGUAGE_OPTIONS } from './lib/i18n';

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

function nodeIdentity(node) {
  if (!node) return '';
  return [node.name || node.id, node.type ? `(${node.type})` : ''].filter(Boolean).join(' ');
}

function nodeDescription(node) {
  if (!node) return '';
  const traits = Array.isArray(node.attributes?.traits) && node.attributes.traits.length
    ? `traits=${node.attributes.traits.join(', ')}`
    : '';
  const parts = [
    node.description || node.short_gloss || '',
    traits,
    `importance=${Number(node.importance ?? 0).toFixed(2)}`,
    `confidence=${Number(node.confidence ?? 0).toFixed(2)}`,
    `frequency=${Number(node.frequency ?? 0).toFixed(0)}`,
  ].filter(Boolean);
  return parts.join(' | ');
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

  const [personalities, setPersonalities] = useState([]);
  const [selectedPersonality, setSelectedPersonality] = useState('');
  const [uploadingFiles, setUploadingFiles] = useState(false);

  const [graphData, setGraphData] = useState({ nodes: [], edges: [], seed_node_ids: [], query: '' });
  const [graphQuery, setGraphQuery] = useState('');
  const [rebuildingGraph, setRebuildingGraph] = useState(false);
  const [selection, setSelection] = useState(null);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(LANGUAGE_STORAGE_KEY, uiLanguage);
    }
  }, [uiLanguage]);

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

  async function loadGraph(query) {
    const result = query ? await getCognitiveGraphSubgraph(query, 16) : await getCognitiveGraph();
    setGraphData({
      nodes: result.nodes || [],
      edges: result.edges || [],
      seed_node_ids: result.seed_node_ids || [],
      query: result.query || query || '',
    });
    return result;
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
      await uploadCognitiveFiles(sessionId, files);
      await loadSession(sessionId);
      await loadGraph(graphQuery.trim() || selectedPersonality || activeSession?.title || files[0].name || '');
      setWorkspace('graph');
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

  const loops = useMemo(() => buildLoopSets(graphData.edges || []), [graphData.edges]);
  const nodesById = useMemo(() => new Map((graphData.nodes || []).map((node) => [String(node.id), node])), [graphData.nodes]);
  const selectedNode = selection?.kind === 'node' ? selection.payload : null;
  const selectedNodeRelations = useMemo(() => nodeRelations(selectedNode?.id, graphData.edges || [], nodesById), [selectedNode, graphData.edges, nodesById]);

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

          {workspace === 'chat' ? (
            <ChatGraphPanel
              session={activeSession}
              value={chatInput}
              onChange={setChatInput}
              running={chatRunning}
              progress={chatProgress}
              onRun={() => void handleRunChat()}
              t={t}
            />
          ) : (
            <div className="graph-shell">
              <div className="graph-toolbar glass-panel">
                <label className="field-stack compact grow">
                  <span>{t('graph_query')}</span>
                  <input value={graphQuery} onChange={(event) => setGraphQuery(event.target.value)} placeholder={t('graph_query_placeholder')} />
                </label>
                <button type="button" onClick={() => void handleGraphSearch()}>{t('graph_search')}</button>
                <button type="button" onClick={() => void handleGraphRebuild()} disabled={rebuildingGraph}>
                  {rebuildingGraph ? t('graph_rebuilding') : t('graph_rebuild')}
                </button>
              </div>
              <GraphWorkspace
                nodes={graphData.nodes || []}
                edges={graphData.edges || []}
                loops={loops}
                selectedNodeId={selection?.kind === 'node' ? selection.id : ''}
                selectedEdgeKey={selection?.kind === 'edge' ? selection.id : ''}
                highlightedNodeIds={new Set(graphData.seed_node_ids || [])}
                highlightedEdgeKeys={new Set()}
                searchHitIds={new Set(graphData.seed_node_ids || [])}
                onSelectNode={(node) => setSelection({ kind: 'node', id: node.id, payload: node })}
                onSelectEdge={(edge) => setSelection({ kind: 'edge', id: `${edge.from || edge.src_id}|${edge.type}|${edge.to || edge.dst_id}`, payload: edge })}
                onClearSelection={() => setSelection(null)}
                t={t}
              />
              <section className="workspace-panel glass-panel node-answer-panel">
                <header className="panel-heading compact">
                  <div>
                    <p className="eyebrow">Graph node</p>
                    <h2>{selectedNode ? (selectedNode.name || selectedNode.id) : t('graph_no_selection')}</h2>
                  </div>
                </header>
                {selectedNode ? (
                  <div className="node-answer-grid">
                    <section>
                      <h3>{t('graph_node_identity')}</h3>
                      <p>{nodeIdentity(selectedNode)}</p>
                    </section>
                    <section>
                      <h3>{t('graph_node_description')}</h3>
                      <p>{nodeDescription(selectedNode)}</p>
                    </section>
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
          )}
        </section>
      </div>
    </main>
  );
}

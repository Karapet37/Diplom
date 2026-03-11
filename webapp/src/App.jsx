import React, { useEffect, useMemo, useState } from "react";
import {
  createCognitiveEdge,
  createCognitiveNode,
  createCognitiveSession,
  getCognitiveGraphSubgraph,
  getCognitiveHealth,
  getCognitiveSession,
  getCognitiveStyleProfile,
  getControlState,
  getHealth,
  getStatus,
  learnCognitiveStyle,
  listCognitiveSessions,
  loadCognitiveFoundation,
  respondCognitiveChat,
  saveCognitiveSession,
  updateCognitiveEdge,
  updateCognitiveNode,
} from "./api";
import { ChatGraphPanel } from "./components/Chat/ChatGraphPanel";
import { GraphEditorPanel } from "./components/Editor/GraphEditorPanel";
import { GraphWorkspace } from "./components/Graph/GraphWorkspace";
import { Sidebar } from "./components/Layout/Sidebar";
import { TopBar } from "./components/Layout/TopBar";
import { createTranslator, LANGUAGE_OPTIONS } from "./lib/i18n";

const LANGUAGE_STORAGE_KEY = "workspace_ui_language_v3";

const EMPTY_TOOLS = {
  user_id: "local_user",
  extra_context: "",
};

const EMPTY_SUBGRAPH = {
  nodes: [],
  edges: [],
  seed_node_ids: [],
  query: "",
};

function nowStamp() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function linesToList(value) {
  return String(value || "")
    .split(/\n|,/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function buildLoopSets(edges) {
  const pairCounts = new Set();
  const twoCycleKeys = new Set();
  const loopNodeIds = new Set();
  for (const edge of edges || []) {
    const src = String(edge.src_id || "");
    const dst = String(edge.dst_id || "");
    if (!src || !dst) continue;
    if (src === dst) {
      loopNodeIds.add(src);
      continue;
    }
    pairCounts.add(`${src}|${dst}`);
  }
  for (const edge of edges || []) {
    const src = String(edge.src_id || "");
    const dst = String(edge.dst_id || "");
    if (!src || !dst || src === dst) continue;
    if (pairCounts.has(`${dst}|${src}`)) {
      loopNodeIds.add(src);
      loopNodeIds.add(dst);
      twoCycleKeys.add(`${src}|${dst}`);
      twoCycleKeys.add(`${dst}|${src}`);
    }
  }
  return { loopNodeIds, twoCycleKeys };
}

function buildNodeDraft(node) {
  if (!node) {
    return {
      id: "",
      type: "",
      label: "",
      short_gloss: "",
      what_it_is: "",
      how_it_works: "",
      how_to_recognize: "",
      examplesText: "",
      tagsText: "",
    };
  }
  return {
    id: String(node.id || ""),
    type: String(node.type || ""),
    label: String(node.label || node.name || ""),
    short_gloss: String(node.short_gloss || node.description || ""),
    what_it_is: String(node.what_it_is || node.plain_explanation || ""),
    how_it_works: String(node.how_it_works || ""),
    how_to_recognize: String(node.how_to_recognize || ""),
    examplesText: Array.isArray(node.examples) ? node.examples.join("\n") : "",
    tagsText: Array.isArray(node.tags) ? node.tags.join(", ") : "",
  };
}

function buildEdgeDraft(edge) {
  if (!edge) {
    return { src_id: "", dst_id: "", type: "", weight: 1, confidence: 0.7 };
  }
  return {
    src_id: String(edge.src_id || ""),
    dst_id: String(edge.dst_id || ""),
    type: String(edge.type || ""),
    weight: Number(edge.weight ?? 1),
    confidence: Number(edge.confidence ?? 0.7),
  };
}

export default function App() {
  const [uiLanguage, setUiLanguage] = useState(() => {
    if (typeof window === "undefined") return "en";
    return window.localStorage.getItem(LANGUAGE_STORAGE_KEY) || "en";
  });
  const t = useMemo(() => createTranslator(uiLanguage), [uiLanguage]);

  const [workspace, setWorkspace] = useState("chat");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const [health, setHealth] = useState(null);
  const [status, setStatus] = useState(null);
  const [controlState, setControlState] = useState(null);
  const [cognitiveHealth, setCognitiveHealth] = useState(null);

  const [sessions, setSessions] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState("");
  const [activeSession, setActiveSession] = useState(null);
  const [sessionTools, setSessionTools] = useState(EMPTY_TOOLS);

  const [chatInput, setChatInput] = useState("");
  const [chatRunning, setChatRunning] = useState(false);
  const [chatProgress, setChatProgress] = useState("");
  const [styleLearning, setStyleLearning] = useState(false);
  const [styleProfile, setStyleProfile] = useState(null);

  const [graphQuery, setGraphQuery] = useState("");
  const [subgraph, setSubgraph] = useState(EMPTY_SUBGRAPH);
  const [subgraphLoading, setSubgraphLoading] = useState(false);
  const [selection, setSelection] = useState(null);
  const [nodeDraft, setNodeDraft] = useState(buildNodeDraft(null));
  const [edgeDraft, setEdgeDraft] = useState(buildEdgeDraft(null));
  const [createNodeDraft, setCreateNodeDraft] = useState({
    node_id: "",
    type: "CONCEPT",
    label: "",
    name: "",
    short_gloss: "",
    description: "",
    what_it_is: "",
    plain_explanation: "",
  });
  const [relationDraft, setRelationDraft] = useState({
    src_id: "",
    dst_id: "",
    type: "RELATED_TO",
    weight: 1,
    confidence: 0.7,
  });
  const [savingNode, setSavingNode] = useState(false);
  const [savingEdge, setSavingEdge] = useState(false);
  const [creatingNode, setCreatingNode] = useState(false);
  const [creatingRelation, setCreatingRelation] = useState(false);
  const [seedRunning, setSeedRunning] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    if (typeof window !== "undefined") {
      window.localStorage.setItem(LANGUAGE_STORAGE_KEY, uiLanguage);
    }
  }, [uiLanguage]);

  useEffect(() => {
    let active = true;
    const boot = async () => {
      setLoading(true);
      try {
        const [nextHealth, nextStatus, nextControlState, nextCognitiveHealth, sessionResult] = await Promise.all([
          getHealth(),
          getStatus(),
          getControlState(),
          getCognitiveHealth(),
          listCognitiveSessions(),
        ]);
        if (!active) return;
        setHealth(nextHealth);
        setStatus(nextStatus);
        setControlState(nextControlState);
        setCognitiveHealth(nextCognitiveHealth);
        const nextSessions = sessionResult.sessions || [];
        setSessions(nextSessions);
        if (nextSessions.length) {
          await loadSession(nextSessions[0].session_id, { suppressError: true });
        } else {
          await createSession();
        }
      } catch (bootError) {
        if (active) {
          setError(bootError.message || String(bootError));
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    };
    void boot();
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (selection?.kind === "node") {
      setNodeDraft(buildNodeDraft(selection.payload));
      setRelationDraft((current) => ({ ...current, src_id: String(selection.id || "") }));
      return;
    }
    if (selection?.kind === "edge") {
      setEdgeDraft(buildEdgeDraft(selection.payload));
      return;
    }
    setNodeDraft(buildNodeDraft(null));
    setEdgeDraft(buildEdgeDraft(null));
  }, [selection]);

  useEffect(() => {
    const query = String(graphQuery || "").trim();
    if (!query) {
      setSubgraph(EMPTY_SUBGRAPH);
      return;
    }
    let active = true;
    setSubgraphLoading(true);
    void getCognitiveGraphSubgraph({ query, limit: 32, hops: 1 })
      .then((result) => {
        if (!active) return;
        setSubgraph({
          nodes: result.nodes || [],
          edges: result.edges || [],
          seed_node_ids: result.seed_node_ids || [],
          query: result.query || query,
        });
      })
      .catch((subgraphError) => {
        if (active) {
          setError(subgraphError.message || String(subgraphError));
        }
      })
      .finally(() => {
        if (active) {
          setSubgraphLoading(false);
        }
      });
    return () => {
      active = false;
    };
  }, [graphQuery]);

  const loopSets = useMemo(() => buildLoopSets(subgraph.edges), [subgraph.edges]);

  async function refreshSystem() {
    setRefreshing(true);
    try {
      const [nextHealth, nextStatus, nextControlState, nextCognitiveHealth, sessionResult] = await Promise.all([
        getHealth(),
        getStatus(),
        getControlState(),
        getCognitiveHealth(),
        listCognitiveSessions(),
      ]);
      setHealth(nextHealth);
      setStatus(nextStatus);
      setControlState(nextControlState);
      setCognitiveHealth(nextCognitiveHealth);
      setSessions(sessionResult.sessions || []);
    } catch (refreshError) {
      setError(refreshError.message || String(refreshError));
    } finally {
      setRefreshing(false);
    }
  }

  async function createSession() {
    try {
      const result = await createCognitiveSession({ title: "New session", tools: sessionTools });
      const session = result.session;
      setSessions((current) => [summarizeSession(session), ...current.filter((item) => item.session_id !== session.session_id)]);
      setActiveSessionId(session.session_id);
      setActiveSession(session);
      setSessionTools({ ...EMPTY_TOOLS, ...(session.tools || {}) });
      setGraphQuery("");
      setWorkspace("chat");
      return session;
    } catch (createError) {
      setError(createError.message || String(createError));
      return null;
    }
  }

  async function loadSession(sessionId, options = {}) {
    try {
      const result = await getCognitiveSession(sessionId);
      const session = result.session;
      setActiveSessionId(session.session_id);
      setActiveSession(session);
      setSessionTools({ ...EMPTY_TOOLS, ...(session.tools || {}) });
      setGraphQuery(session.last_query || "");
      if (session.tools?.user_id) {
        try {
          const profile = await getCognitiveStyleProfile(session.tools.user_id);
          setStyleProfile(profile.profile || null);
        } catch (_ignored) {
          setStyleProfile(null);
        }
      } else {
        setStyleProfile(null);
      }
    } catch (sessionError) {
      if (!options.suppressError) {
        setError(sessionError.message || String(sessionError));
      }
    }
  }

  async function persistSession(nextSession) {
    if (!nextSession?.session_id) return;
    const result = await saveCognitiveSession(nextSession.session_id, {
      title: nextSession.title,
      last_query: nextSession.last_query || "",
      tools: nextSession.tools || {},
      messages: nextSession.messages || [],
    });
    const saved = result.session;
    setActiveSession(saved);
    setSessions((current) => [summarizeSession(saved), ...current.filter((item) => item.session_id !== saved.session_id)]);
  }

  async function runChat() {
    if (!chatInput.trim()) return;
    setError("");
    setChatRunning(true);
    setChatProgress(t("chat_thinking"));
    const baseSession = activeSession?.session_id ? activeSession : await createSession();
    if (!baseSession?.session_id) {
      setChatRunning(false);
      setChatProgress("");
      return;
    }

    const userMessage = {
      id: `msg-user-${Date.now()}`,
      role: "user",
      message: chatInput.trim(),
      timestamp: nowStamp(),
    };
    const placeholder = {
      id: `msg-assistant-${Date.now()}`,
      role: "assistant",
      message: t("chat_thinking"),
      timestamp: nowStamp(),
      metadata: {},
    };
    const optimisticSession = {
      ...baseSession,
      title: baseSession.title || buildSessionTitle(chatInput),
      last_query: chatInput.trim(),
      tools: sessionTools,
      messages: [...(baseSession.messages || []), userMessage, placeholder],
    };
    setActiveSession(optimisticSession);
    setSessions((current) => [summarizeSession(optimisticSession), ...current.filter((item) => item.session_id !== optimisticSession.session_id)]);
    const outgoing = chatInput;
    setChatInput("");

    try {
      const result = await respondCognitiveChat({
        message: outgoing,
        context: sessionTools.extra_context || "",
        user_id: sessionTools.user_id || "",
        save_to_graph: true,
        apply_to_graph: true,
      });
      const assistantMessage = {
        id: `msg-assistant-${Date.now()}-done`,
        role: "assistant",
        message: result.assistant_reply || t("chat_no_reply"),
        timestamp: nowStamp(),
        metadata: {
          graphDiff: result.graph_diff || null,
          graphJob: result.graph_job || null,
          contextNodes: result.context_nodes || result.ram_graph?.ranked_context || [],
          ramGraph: result.ram_graph || {},
          style: result.style || null,
        },
      };
      const savedSession = {
        ...optimisticSession,
        messages: [...(baseSession.messages || []), userMessage, assistantMessage],
      };
      await persistSession(savedSession);
      setGraphQuery(outgoing);
    } catch (runError) {
      setError(runError.message || String(runError));
      const failedSession = {
        ...baseSession,
        messages: [...(baseSession.messages || []), userMessage],
      };
      setActiveSession(failedSession);
      await persistSession({ ...failedSession, tools: sessionTools, last_query: outgoing });
    } finally {
      setChatRunning(false);
      setChatProgress("");
    }
  }

  async function runLearnStyle() {
    const userId = String(sessionTools.user_id || "").trim();
    const session = activeSession;
    if (!userId || !session?.messages?.length) return;
    setStyleLearning(true);
    try {
      const result = await learnCognitiveStyle({
        user_id: userId,
        learn_style_button: true,
        messages: session.messages.map((item) => ({ role: item.role, message: item.message })),
        max_messages: 12,
      });
      setStyleProfile(result.profile || null);
    } catch (styleError) {
      setError(styleError.message || String(styleError));
    } finally {
      setStyleLearning(false);
    }
  }

  async function loadFoundation(datasetId) {
    setSeedRunning(true);
    try {
      await loadCognitiveFoundation({ dataset_id: datasetId, replace_graph: true });
      if (graphQuery.trim()) {
        const result = await getCognitiveGraphSubgraph({ query: graphQuery.trim(), limit: 32, hops: 1 });
        setSubgraph({
          nodes: result.nodes || [],
          edges: result.edges || [],
          seed_node_ids: result.seed_node_ids || [],
          query: result.query || graphQuery.trim(),
        });
      }
    } catch (seedError) {
      setError(seedError.message || String(seedError));
    } finally {
      setSeedRunning(false);
    }
  }

  async function saveSelectedNode() {
    if (!nodeDraft.id) return;
    setSavingNode(true);
    try {
      const result = await updateCognitiveNode(nodeDraft.id, {
        type: nodeDraft.type,
        label: nodeDraft.label,
        short_gloss: nodeDraft.short_gloss,
        what_it_is: nodeDraft.what_it_is,
        how_it_works: nodeDraft.how_it_works,
        how_to_recognize: nodeDraft.how_to_recognize,
        examples: linesToList(nodeDraft.examplesText),
        tags: linesToList(nodeDraft.tagsText),
      });
      setSelection({ kind: "node", id: nodeDraft.id, payload: result.node });
      if (graphQuery.trim()) {
        const refreshed = await getCognitiveGraphSubgraph({ query: graphQuery.trim(), limit: 32, hops: 1 });
        setSubgraph({
          nodes: refreshed.nodes || [],
          edges: refreshed.edges || [],
          seed_node_ids: refreshed.seed_node_ids || [],
          query: refreshed.query || graphQuery.trim(),
        });
      }
    } catch (saveError) {
      setError(saveError.message || String(saveError));
    } finally {
      setSavingNode(false);
    }
  }

  async function saveSelectedEdge() {
    if (!edgeDraft.src_id || !edgeDraft.dst_id || !edgeDraft.type) return;
    setSavingEdge(true);
    try {
      const result = await updateCognitiveEdge(edgeDraft);
      setSelection({ kind: "edge", id: `${edgeDraft.src_id}|${edgeDraft.type}|${edgeDraft.dst_id}`, payload: result.edge });
      if (graphQuery.trim()) {
        const refreshed = await getCognitiveGraphSubgraph({ query: graphQuery.trim(), limit: 32, hops: 1 });
        setSubgraph({
          nodes: refreshed.nodes || [],
          edges: refreshed.edges || [],
          seed_node_ids: refreshed.seed_node_ids || [],
          query: refreshed.query || graphQuery.trim(),
        });
      }
    } catch (saveError) {
      setError(saveError.message || String(saveError));
    } finally {
      setSavingEdge(false);
    }
  }

  async function createNode() {
    if (!createNodeDraft.node_id || !createNodeDraft.label) return;
    setCreatingNode(true);
    try {
      await createCognitiveNode(createNodeDraft);
      const nextQuery = graphQuery.trim() || createNodeDraft.label;
      setGraphQuery(nextQuery);
      setCreateNodeDraft({
        node_id: "",
        type: "CONCEPT",
        label: "",
        name: "",
        short_gloss: "",
        description: "",
        what_it_is: "",
        plain_explanation: "",
      });
    } catch (createError) {
      setError(createError.message || String(createError));
    } finally {
      setCreatingNode(false);
    }
  }

  async function createRelation() {
    if (!relationDraft.src_id || !relationDraft.dst_id || !relationDraft.type) return;
    setCreatingRelation(true);
    try {
      await createCognitiveEdge(relationDraft);
      if (graphQuery.trim()) {
        const refreshed = await getCognitiveGraphSubgraph({ query: graphQuery.trim(), limit: 32, hops: 1 });
        setSubgraph({
          nodes: refreshed.nodes || [],
          edges: refreshed.edges || [],
          seed_node_ids: refreshed.seed_node_ids || [],
          query: refreshed.query || graphQuery.trim(),
        });
      }
      setRelationDraft((current) => ({ ...current, dst_id: "", type: "RELATED_TO", weight: 1, confidence: 0.7 }));
    } catch (createError) {
      setError(createError.message || String(createError));
    } finally {
      setCreatingRelation(false);
    }
  }

  const selectionLabel = selection?.kind === "node"
    ? (selection.payload?.label || selection.payload?.name || selection.id)
    : selection?.kind === "edge"
      ? `${selection.payload?.src_id || ""} → ${selection.payload?.dst_id || ""}`
      : "";

  if (loading) {
    return (
      <div className="operating-room loading-screen">
        <div className="loading-card glass-panel">
          <p className="eyebrow">{t("common_loading")}</p>
          <h1>{t("top_title")}</h1>
        </div>
      </div>
    );
  }

  return (
    <div className="operating-room">
      <TopBar
        health={health}
        engineHealth={cognitiveHealth}
        controlState={controlState}
        searchTerm={graphQuery}
        onSearchChange={setGraphQuery}
        language={uiLanguage}
        languageOptions={LANGUAGE_OPTIONS}
        onLanguageChange={setUiLanguage}
        onRefresh={() => void refreshSystem()}
        onRebuild={null}
        onRunQuality={null}
        refreshing={refreshing}
        rebuilding={false}
        qualityRunning={false}
        t={t}
      />

      {error ? (
        <section className="glass-panel error-banner">
          <strong>{t("runtime_error")}</strong>
          <span>{error}</span>
        </section>
      ) : null}

      <div className="operating-room__body workspace-shell">
        <Sidebar
          workspace={workspace}
          onWorkspaceChange={setWorkspace}
          sessions={sessions}
          activeSessionId={activeSessionId}
          onSelectSession={(sessionId) => void loadSession(sessionId)}
          onCreateSession={() => void createSession()}
          sessionTools={sessionTools}
          onSessionToolsChange={(tools) => {
            setSessionTools(tools);
            if (activeSession?.session_id) {
              void persistSession({ ...activeSession, tools });
            }
          }}
          onLearnStyle={() => void runLearnStyle()}
          styleLearning={styleLearning}
          styleProfile={styleProfile}
          graphQuery={graphQuery}
          onGraphQueryChange={setGraphQuery}
          onUseSessionQuery={() => setGraphQuery(activeSession?.last_query || "")}
          onQuickLoadHuman={() => void loadFoundation("human_foundations")}
          onQuickLoadPsychology={() => void loadFoundation("psychology_foundations")}
          seedRunning={seedRunning}
          t={t}
        />

        <div className="workspace-center workspace-center--single">
          {workspace === "chat" ? (
            <ChatGraphPanel
              session={activeSession}
              value={chatInput}
              onChange={setChatInput}
              running={chatRunning}
              progress={chatProgress}
              onRun={() => void runChat()}
              t={t}
            />
          ) : (
            <div className="graph-workspace-shell">
              <section className="graph-stage glass-panel">
                <header className="panel-heading compact">
                  <div>
                    <p className="eyebrow">{t("graph_body")}</p>
                    <h2>{t("sidebar_graph_workspace")}</h2>
                    <p>{graphQuery ? `${t("graph_query_label")}: ${graphQuery}` : t("graph_workspace_empty_text")}</p>
                  </div>
                </header>
                <GraphWorkspace
                  nodes={subgraph.nodes}
                  edges={subgraph.edges}
                  loops={loopSets}
                  selectedNodeId={selection?.kind === "node" ? selection.id : ""}
                  selectedEdgeKey={selection?.kind === "edge" ? selection.id : ""}
                  highlightedNodeIds={new Set(subgraph.seed_node_ids || [])}
                  highlightedEdgeKeys={new Set()}
                  searchHitIds={new Set(subgraph.seed_node_ids || [])}
                  onSelectNode={(node) => setSelection({ kind: "node", id: String(node.id), payload: node })}
                  onSelectEdge={(edge) => setSelection({ kind: "edge", id: edge.edge_key || `${edge.src_id}|${edge.type}|${edge.dst_id}`, payload: edge })}
                  onClearSelection={() => setSelection(null)}
                  t={t}
                />
                {subgraphLoading ? <div className="graph-inline-status">{t("top_refreshing")}</div> : null}
              </section>

              <GraphEditorPanel
                selection={selection}
                nodeDraft={nodeDraft}
                edgeDraft={edgeDraft}
                createNodeDraft={createNodeDraft}
                relationDraft={relationDraft}
                visibleNodes={subgraph.nodes}
                onNodeDraftChange={setNodeDraft}
                onEdgeDraftChange={setEdgeDraft}
                onCreateNodeDraftChange={setCreateNodeDraft}
                onRelationDraftChange={setRelationDraft}
                onSaveNode={() => void saveSelectedNode()}
                onSaveEdge={() => void saveSelectedEdge()}
                onCreateNode={() => void createNode()}
                onCreateRelation={() => void createRelation()}
                savingNode={savingNode}
                savingEdge={savingEdge}
                creatingNode={creatingNode}
                creatingRelation={creatingRelation}
                t={t}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function summarizeSession(session) {
  return {
    session_id: session.session_id,
    title: session.title || "Untitled session",
    created_at: session.created_at || "",
    updated_at: session.updated_at || "",
    last_query: session.last_query || "",
    message_count: Array.isArray(session.messages) ? session.messages.length : 0,
  };
}

function buildSessionTitle(message) {
  const raw = String(message || "").trim();
  if (!raw) return "New session";
  return raw.length > 48 ? `${raw.slice(0, 48)}…` : raw;
}

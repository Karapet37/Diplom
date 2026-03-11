const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      if (payload && payload.detail) {
        detail = typeof payload.detail === "string" ? payload.detail : JSON.stringify(payload.detail);
      }
    } catch (_error) {
      // ignore
    }
    throw new Error(detail);
  }
  return response.json();
}

async function requestAt(prefix, path, options = {}) {
  const normalizedPrefix = String(prefix || "").replace(/\/+$/, "");
  const normalizedPath = String(path || "").startsWith("/") ? path : `/${path}`;
  return request(`${normalizedPrefix}${normalizedPath}`, options);
}

function resolveWebSocketUrl(path) {
  const base = String(API_BASE || "").trim();

  if (base.startsWith("ws://") || base.startsWith("wss://")) {
    const normalizedBase = base.replace(/\/+$/, "");
    return `${normalizedBase}${path}`;
  }

  const origin = typeof window !== "undefined" ? window.location.origin : "http://127.0.0.1";
  if (base.startsWith("http://") || base.startsWith("https://")) {
    const url = new URL(path, base);
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
    return url.toString();
  }

  const url = new URL(path, origin);
  if (base) {
    const normalizedBase = `/${base}`.replace(/\/+/g, "/").replace(/\/$/, "");
    url.pathname = `${normalizedBase}${path}`.replace(/\/+/g, "/");
  }
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  return url.toString();
}

export function subscribeGraphEvents(handlers = {}) {
  const {
    onOpen = () => {},
    onMessage = () => {},
    onClose = () => {},
    onError = () => {},
  } = handlers;

  const socket = new WebSocket(resolveWebSocketUrl("/api/graph/ws"));
  socket.addEventListener("open", () => onOpen());
  socket.addEventListener("message", (event) => {
    try {
      const payload = JSON.parse(String(event?.data || ""));
      onMessage(payload);
    } catch (_error) {
      // ignore malformed frame
    }
  });
  socket.addEventListener("error", (event) => onError(event));
  socket.addEventListener("close", () => onClose());
  return () => {
    try {
      if (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING) {
        socket.close();
      }
    } catch (_error) {
      // ignore
    }
  };
}

export function getHealth() {
  return request("/api/health");
}

export function getStatus() {
  return request("/api/status");
}

export function getControlState() {
  return request("/api/control/state");
}

export function getModules() {
  return request("/api/modules");
}

export function getProjectOverview() {
  return request("/api/project/overview");
}

export function getNodeTypes() {
  return request("/api/graph/node-types");
}

export function getSnapshot() {
  return request("/api/graph/snapshot");
}

export function getEvents(limit = 200, eventType = "") {
  const query = new URLSearchParams({ limit: String(limit) });
  if (eventType) {
    query.set("event_type", eventType);
  }
  return request(`/api/graph/events?${query.toString()}`);
}

export function createNode(payload) {
  return request("/api/graph/node", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function createEdge(payload) {
  return request("/api/graph/edge", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function updateNode(payload) {
  return request("/api/graph/node/update", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function createGraphFoundation(payload = {}) {
  return request("/api/graph/foundation/create", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function runGraphNodeAssist(payload = {}) {
  return request("/api/graph/node/assist", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function runGraphEdgeAssist(payload = {}) {
  return request("/api/graph/edge/assist", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function deleteNode(payload) {
  return request("/api/graph/node/delete", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function updateEdge(payload) {
  return request("/api/graph/edge/update", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function deleteEdge(payload) {
  return request("/api/graph/edge/delete", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function simulateGraph(payload) {
  return request("/api/graph/simulate", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function rewardEvent(payload) {
  return request("/api/graph/event/reward", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function reinforceRelation(payload) {
  return request("/api/graph/relation/reinforce", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function persistGraph() {
  return request("/api/graph/persist", { method: "POST" });
}

export function loadGraph() {
  return request("/api/graph/load", { method: "POST" });
}

export function clearGraph() {
  return request("/api/graph/clear", { method: "POST" });
}

export function seedDemoGraph() {
  return request("/api/graph/seed-demo", { method: "POST" });
}

export function watchProjectDemo(payload = {}) {
  return request("/api/project/demo/watch", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function runProjectDailyMode(payload = {}) {
  return request("/api/project/daily-mode", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function runProjectLLMDebate(payload = {}) {
  return request("/api/project/llm/debate", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function runProjectArchiveChat(payload = {}) {
  return request("/api/project/archive/chat", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function runProjectChatGraph(payload = {}) {
  return request("/api/project/chat-graph", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getProjectWikipediaRandom(language = "ru") {
  const query = new URLSearchParams({ language: String(language || "ru") });
  return request(`/api/project/wiki/random?${query.toString()}`);
}

export function searchProjectWikipedia(queryText = "", language = "ru", limit = 5) {
  const query = new URLSearchParams({
    query: String(queryText || ""),
    language: String(language || "ru"),
    limit: String(limit || 5),
  });
  return request(`/api/project/wiki/search?${query.toString()}`);
}

export function resolveProjectModePolicy(payload = {}) {
  return request("/api/project/mode-policy/resolve", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function saveProjectContextMode(payload = {}) {
  return request("/api/project/mode/save", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function captureProjectContextFocus(payload = {}) {
  return request("/api/project/mode/focus", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function feedbackProjectContextMode(payload = {}) {
  return request("/api/project/mode/feedback", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function applyProjectArchiveReview(payload = {}) {
  return request("/api/project/archive/review", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function reportProjectHallucination(payload = {}) {
  return request("/api/project/hallucination/report", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function checkProjectHallucination(payload = {}) {
  return request("/api/project/hallucination/check", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function updateProjectUserGraph(payload = {}) {
  return request("/api/project/user-graph/update", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function runProjectPersonalTreeIngest(payload = {}) {
  return request("/api/project/personal-tree/ingest", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function saveProjectPersonalTreeNote(payload = {}) {
  return request("/api/project/personal-tree/note", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function viewProjectPersonalTree(payload = {}) {
  return request("/api/project/personal-tree/view", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function manageProjectPackages(payload = {}) {
  return request("/api/project/packages/manage", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function applyProjectMemoryNamespace(payload = {}) {
  return request("/api/project/memory/namespace/apply", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function viewProjectMemoryNamespace(payload = {}) {
  return request("/api/project/memory/namespace/view", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function runProjectGraphRagQuery(payload = {}) {
  return request("/api/project/graph-rag/query", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function scanProjectContradictions(payload = {}) {
  return request("/api/project/contradiction/scan", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function runProjectTaskRiskBoard(payload = {}) {
  return request("/api/project/task-risk/board", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function runProjectTimelineReplay(payload = {}) {
  return request("/api/project/timeline/replay", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getProjectLLMPolicy() {
  return request("/api/project/llm-policy");
}

export function updateProjectLLMPolicy(payload = {}) {
  return request("/api/project/llm-policy", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function runProjectQualityHarness(payload = {}) {
  return request("/api/project/quality/harness", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function createProjectBackup(payload = {}) {
  return request("/api/project/backup/create", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function restoreProjectBackup(payload = {}) {
  return request("/api/project/backup/restore", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getProjectAuditLogs(payload = {}) {
  return request("/api/project/audit/logs", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function runProjectWrapperRespond(payload = {}) {
  return request("/api/project/wrapper/respond", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getProjectWrapperProfile(userId = "default_user") {
  const query = new URLSearchParams({ user_id: String(userId || "default_user") });
  return request(`/api/project/wrapper/profile?${query.toString()}`);
}

export function updateProjectWrapperProfile(payload = {}) {
  return request("/api/project/wrapper/profile", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function sendProjectWrapperFeedback(payload = {}) {
  return request("/api/project/wrapper/feedback", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getProjectIntegrationLayerManifest(params = {}) {
  const query = new URLSearchParams();
  const host = String(params?.host || "generic").trim();
  const appId = String(params?.app_id || params?.appId || "external_app").trim();
  if (host) {
    query.set("host", host);
  }
  if (appId) {
    query.set("app_id", appId);
  }
  const suffix = query.toString();
  return request(`/api/integration/layer/manifest${suffix ? `?${suffix}` : ""}`);
}

export function invokeProjectIntegrationLayer(payload = {}) {
  return request("/api/integration/layer/invoke", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function introspectClient(payload = {}) {
  return request("/api/client/introspect", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getProjectDbSchema() {
  return request("/api/project/db/schema");
}

export function getProjectModelAdvisors() {
  return request("/api/project/model-advisors");
}

export function getCognitiveHealth() {
  return requestAt("/api/cognitive", "/health");
}

export function getCognitiveGraph(params = {}) {
  const query = new URLSearchParams();
  if (params.edge_type) {
    query.set("edge_type", String(params.edge_type));
  }
  if (params.min_weight != null) {
    query.set("min_weight", String(params.min_weight));
  }
  const suffix = query.toString();
  return requestAt("/api/cognitive", `/graph${suffix ? `?${suffix}` : ""}`);
}

export function getCognitiveGraphSubgraph(params = {}) {
  const query = new URLSearchParams();
  if (params.query) {
    query.set("query", String(params.query));
  }
  if (params.limit != null) {
    query.set("limit", String(params.limit));
  }
  if (params.hops != null) {
    query.set("hops", String(params.hops));
  }
  return requestAt("/api/cognitive", `/graph/subgraph?${query.toString()}`);
}

export function getCognitiveLoops() {
  return requestAt("/api/cognitive", "/analysis/loops");
}

export function getCognitiveGraphAudit() {
  return requestAt("/api/cognitive", "/graph/audit");
}

export function getCognitiveFoundations() {
  return requestAt("/api/cognitive", "/foundations");
}


export function getCognitiveSources() {
  return requestAt("/api/cognitive", "/sources");
}

export function listCognitiveSessions() {
  return requestAt("/api/cognitive", "/sessions");
}

export function createCognitiveSession(payload = {}) {
  return requestAt("/api/cognitive", "/sessions", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getCognitiveSession(sessionId) {
  return requestAt("/api/cognitive", `/sessions/${encodeURIComponent(String(sessionId || ""))}`);
}

export function saveCognitiveSession(sessionId, payload = {}) {
  return requestAt("/api/cognitive", `/sessions/${encodeURIComponent(String(sessionId || ""))}`, {
    method: "PUT",
    body: JSON.stringify(payload),
  });
}

export function ingestCognitiveText(payload = {}) {
  return requestAt("/api/cognitive", "/ingest", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function rebuildCognitiveGraph(payload = {}) {
  return requestAt("/api/cognitive", "/rebuild", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function interpretCognitiveText(payload = {}) {
  return requestAt("/api/cognitive", "/interpret", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function planCognitive(payload = {}) {
  return requestAt("/api/cognitive", "/plan", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function loadCognitiveFoundation(payload = {}) {
  return requestAt("/api/cognitive", "/foundations/load", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function importCognitiveGraph(payload = {}) {
  return requestAt("/api/cognitive", "/graph/import", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function respondCognitiveChat(payload = {}) {
  return requestAt("/api/cognitive", "/chat/respond", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function learnCognitiveStyle(payload = {}) {
  return requestAt("/api/cognitive", "/style/learn", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getCognitiveStyleProfile(userId) {
  const query = new URLSearchParams({ user_id: String(userId || "") });
  return requestAt("/api/cognitive", `/style/profile?${query.toString()}`);
}

export function updateCognitiveNode(nodeId, payload = {}) {
  return requestAt("/api/cognitive", `/nodes/${encodeURIComponent(String(nodeId || ""))}`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
}

export function createCognitiveNode(payload = {}) {
  return requestAt("/api/cognitive", "/nodes", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function updateCognitiveEdge(payload = {}) {
  return requestAt("/api/cognitive", "/edges", {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
}

export function createCognitiveEdge(payload = {}) {
  return requestAt("/api/cognitive", "/edges", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getProfilePrompt(entityTypeHint = "human") {
  const query = new URLSearchParams({ entity_type_hint: String(entityTypeHint || "human") });
  return request(`/api/graph/profile/prompt?${query.toString()}`);
}

export function inferProfileGraph(payload) {
  return request("/api/graph/profile/infer", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

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

export function getModules() {
  return request("/api/modules");
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

export function importProjectAutoruns(payload = {}) {
  return request("/api/project/autoruns/import", {
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

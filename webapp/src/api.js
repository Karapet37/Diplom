const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      ...(options.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }),
      ...(options.headers || {}),
    },
    ...options,
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      if (payload?.detail) {
        detail = typeof payload.detail === 'string' ? payload.detail : JSON.stringify(payload.detail);
      }
    } catch (_error) {
      // ignore
    }
    throw new Error(detail);
  }
  return response.json();
}

export function getHealth() {
  return request('/api/health');
}

export function listCognitiveSessions() {
  return request('/api/cognitive/sessions');
}

export function createCognitiveSession(payload = {}) {
  return request('/api/cognitive/sessions', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export function getCognitiveSession(sessionId) {
  return request(`/api/cognitive/sessions/${encodeURIComponent(sessionId)}`);
}

export function deleteCognitiveSession(sessionId) {
  return request(`/api/cognitive/sessions/${encodeURIComponent(sessionId)}`, {
    method: 'DELETE',
  });
}

export function respondCognitiveChat(payload = {}) {
  return request('/api/cognitive/chat/respond', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export function listCognitivePersonalities() {
  return request('/api/cognitive/personalities');
}

export function getCognitivePersonality(name) {
  return request(`/api/cognitive/personalities/${encodeURIComponent(name)}`);
}

export function getCognitiveGraph() {
  return request('/api/cognitive/graph');
}

export function getCognitiveGraphSubgraph(query = '', limit = 12) {
  const params = new URLSearchParams();
  if (query) {
    params.set('query', String(query));
  }
  params.set('limit', String(limit));
  return request(`/api/cognitive/graph/subgraph?${params.toString()}`);
}

export function getCognitiveGraphNodeView(nodeId, language = '') {
  const params = new URLSearchParams();
  if (language) {
    params.set('language', String(language));
  }
  const suffix = params.toString() ? `?${params.toString()}` : '';
  return request(`/api/cognitive/graph/nodes/${encodeURIComponent(nodeId)}/view${suffix}`);
}

export function createCognitiveGraphNode(payload = {}) {
  return request('/api/cognitive/graph/nodes', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export function connectCognitiveGraphNodes(payload = {}) {
  return request('/api/cognitive/graph/edges', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export function mergeCognitiveGraphNodes(payload = {}) {
  return request('/api/cognitive/graph/nodes/merge', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export function reviewCognitiveGraphNode(nodeId, payload = {}) {
  return request(`/api/cognitive/graph/nodes/${encodeURIComponent(nodeId)}/state`, {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export function deleteCognitiveGraphNode(nodeId) {
  return request(`/api/cognitive/graph/nodes/${encodeURIComponent(nodeId)}`, {
    method: 'DELETE',
  });
}

export function deleteCognitiveGraphEdge(edgeId) {
  return request(`/api/cognitive/graph/edges/${encodeURIComponent(edgeId)}`, {
    method: 'DELETE',
  });
}

export function rethinkCognitiveGraphNodes(payload = {}) {
  return request('/api/cognitive/graph/rethink', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export function getCognitiveDebugMetrics() {
  return request('/api/cognitive/debug/metrics');
}

export function getCognitiveDebugTraces({ limit = 20, sessionId = '' } = {}) {
  const params = new URLSearchParams();
  params.set('limit', String(limit));
  if (sessionId) {
    params.set('session_id', String(sessionId));
  }
  return request(`/api/cognitive/debug/traces?${params.toString()}`);
}

export function getCognitiveGraphHealth() {
  return request('/api/cognitive/debug/graph-health');
}

export function rebuildCognitiveGraph(payload = {}) {
  return request('/api/cognitive/rebuild', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export function uploadCognitiveFiles(sessionId, files = []) {
  const formData = new FormData();
  formData.append('session_id', sessionId || '');
  for (const file of files) {
    formData.append('files', file);
  }
  return request('/api/cognitive/files/upload', {
    method: 'POST',
    body: formData,
  });
}

// ---------------------------------------------------------------------------
// Personality construction — delete
// ---------------------------------------------------------------------------

export function deletePersonality(personalityId) {
  return request(`/api/cognitive/personality-construction/personalities/${encodeURIComponent(personalityId)}`, {
    method: 'DELETE',
  });
}

// ---------------------------------------------------------------------------
// Training examples — fine-tuning dataset curation
// ---------------------------------------------------------------------------

export function listTrainingExamples({ sessionId = '', personaName = '' } = {}) {
  const params = new URLSearchParams();
  if (sessionId) params.set('session_id', String(sessionId));
  if (personaName) params.set('persona_name', String(personaName));
  const suffix = params.toString() ? `?${params.toString()}` : '';
  return request(`/api/cognitive/training-examples${suffix}`);
}

export function createTrainingExample(payload = {}) {
  return request('/api/cognitive/training-examples', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export function deleteTrainingExample(exampleId) {
  return request(`/api/cognitive/training-examples/${encodeURIComponent(exampleId)}`, {
    method: 'DELETE',
  });
}

export function exportTrainingExamples({ sessionId = '', personaName = '', fmt = 'openai' } = {}) {
  const params = new URLSearchParams();
  if (sessionId) params.set('session_id', String(sessionId));
  if (personaName) params.set('persona_name', String(personaName));
  params.set('fmt', String(fmt));
  return request(`/api/cognitive/training-examples/export/jsonl?${params.toString()}`);
}

// ---------------------------------------------------------------------------
// Safety classifier
// ---------------------------------------------------------------------------

export function classifySafety(text, language = 'en') {
  return request('/api/cognitive/safety/classify', {
    method: 'POST',
    body: JSON.stringify({ text: String(text || ''), language: String(language || 'en') }),
  });
}

export function addSafetyExample(text, label, notes = '') {
  return request('/api/cognitive/safety/examples', {
    method: 'POST',
    body: JSON.stringify({ text: String(text || ''), label: String(label || ''), notes: String(notes || '') }),
  });
}

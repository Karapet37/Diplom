export function parseJsonList(raw) {
  if (!raw) {
    return [];
  }
  if (Array.isArray(raw)) {
    return raw;
  }
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch (_error) {
    return [];
  }
}

export function getNodeLabel(node) {
  if (!node) {
    return "Unknown";
  }
  return (
    node.label ||
    node.name ||
    node.title ||
    node.short_gloss ||
    node.attributes?.title ||
    node.attributes?.name ||
    node.attributes?.label ||
    node.id ||
    "Untitled"
  );
}

export function getNodeGloss(node) {
  if (!node) {
    return "";
  }
  return node.short_gloss || node.attributes?.summary || node.attributes?.description || "";
}

export function getNodeExplanation(node) {
  if (!node) {
    return "";
  }
  return node.plain_explanation || node.attributes?.plain_explanation || node.attributes?.description || "";
}

export function getNodeExamples(node) {
  return parseJsonList(node?.examples_json || node?.attributes?.examples_json || node?.attributes?.examples || []);
}

export function getNodeTags(node) {
  return parseJsonList(node?.tags_json || node?.attributes?.tags_json || node?.attributes?.tags || []);
}

export function getNodeImages(node) {
  const images = [];
  const attrs = node?.attributes || {};
  if (typeof attrs.image === "string" && attrs.image.trim()) {
    images.push(attrs.image.trim());
  }
  if (typeof attrs.image_url === "string" && attrs.image_url.trim()) {
    images.push(attrs.image_url.trim());
  }
  if (Array.isArray(attrs.images)) {
    for (const item of attrs.images) {
      if (typeof item === "string" && item.trim()) {
        images.push(item.trim());
      }
    }
  }
  return [...new Set(images)];
}

export function collectNodeTypes(nodes = []) {
  return [...new Set(nodes.map((node) => String(node.type || "generic")))].sort();
}

export function collectEdgeTypes(edges = []) {
  return [...new Set(edges.map((edge) => String(edge.type || edge.relation_type || "generic")))].sort();
}

export function edgeTypeOf(edge) {
  return String(edge?.type || edge?.relation_type || "generic");
}

export function edgeWeightOf(edge) {
  const value = Number(edge?.weight || 0);
  return Number.isFinite(value) ? value : 0;
}

export function buildAdjacency(edges = []) {
  const adjacency = new Map();
  for (const edge of edges) {
    const src = String(edge.src_id || edge.from || "");
    const dst = String(edge.dst_id || edge.to || "");
    if (!src || !dst) {
      continue;
    }
    adjacency.set(src, [...(adjacency.get(src) || []), dst]);
    adjacency.set(dst, [...(adjacency.get(dst) || []), src]);
  }
  return adjacency;
}

export function bfsNeighborhood(seedId, edges = [], depth = 1) {
  const start = String(seedId || "");
  if (!start) {
    return new Set();
  }
  const adjacency = buildAdjacency(edges);
  const visited = new Set([start]);
  const queue = [{ id: start, level: 0 }];
  while (queue.length) {
    const current = queue.shift();
    if (!current || current.level >= depth) {
      continue;
    }
    for (const next of adjacency.get(current.id) || []) {
      if (visited.has(next)) {
        continue;
      }
      visited.add(next);
      queue.push({ id: next, level: current.level + 1 });
    }
  }
  return visited;
}

export function shortestPathBetween(startId, endId, edges = []) {
  const start = String(startId || "");
  const goal = String(endId || "");
  if (!start || !goal || start === goal) {
    return start && goal ? [start] : [];
  }
  const adjacency = buildAdjacency(edges);
  const queue = [start];
  const previous = new Map([[start, null]]);
  while (queue.length) {
    const current = queue.shift();
    if (current === goal) {
      break;
    }
    for (const next of adjacency.get(current) || []) {
      if (previous.has(next)) {
        continue;
      }
      previous.set(next, current);
      queue.push(next);
    }
  }
  if (!previous.has(goal)) {
    return [];
  }
  const path = [];
  let cursor = goal;
  while (cursor) {
    path.unshift(cursor);
    cursor = previous.get(cursor) || null;
  }
  return path;
}

export function computeLoopSets(loops = {}) {
  const loopNodeIds = new Set();
  const twoCycleKeys = new Set();
  for (const edge of loops.self_loops || []) {
    const nodeId = String(edge.src_id || edge.dst_id || "");
    if (nodeId) {
      loopNodeIds.add(nodeId);
    }
  }
  for (const row of loops.two_cycles || []) {
    const a = String(row.a || "");
    const b = String(row.b || "");
    if (!a || !b) {
      continue;
    }
    twoCycleKeys.add(`${a}|${b}`);
    twoCycleKeys.add(`${b}|${a}`);
    loopNodeIds.add(a);
    loopNodeIds.add(b);
  }
  return { loopNodeIds, twoCycleKeys };
}

export function deterministicLayout(nodes = [], edges = [], width = 1180, height = 780) {
  const centerX = width / 2;
  const centerY = height / 2;
  const groups = new Map();
  const degrees = new Map();
  for (const node of nodes) {
    const type = String(node.type || "generic");
    groups.set(type, [...(groups.get(type) || []), node]);
    degrees.set(String(node.id), 0);
  }
  for (const edge of edges) {
    const left = String(edge.src_id || edge.from || "");
    const right = String(edge.dst_id || edge.to || "");
    if (degrees.has(left)) {
      degrees.set(left, Number(degrees.get(left) || 0) + 1);
    }
    if (degrees.has(right)) {
      degrees.set(right, Number(degrees.get(right) || 0) + 1);
    }
  }

  const orderedGroups = [...groups.entries()].sort((a, b) => b[1].length - a[1].length || a[0].localeCompare(b[0]));
  const positions = {};
  const groupRadius = Math.max(210, Math.min(width, height) * 0.28);

  orderedGroups.forEach(([type, groupNodes], groupIndex) => {
    const sectorAngle = (Math.PI * 2 * groupIndex) / Math.max(orderedGroups.length, 1) - Math.PI / 2;
    const groupCenterX = centerX + Math.cos(sectorAngle) * groupRadius;
    const groupCenterY = centerY + Math.sin(sectorAngle) * groupRadius;
    const sortedNodes = [...groupNodes].sort((left, right) => {
      const degreeDelta = Number(degrees.get(String(right.id)) || 0) - Number(degrees.get(String(left.id)) || 0);
      if (degreeDelta !== 0) {
        return degreeDelta;
      }
      return getNodeLabel(left).localeCompare(getNodeLabel(right));
    });
    const innerRadius = 58 + Math.min(sortedNodes.length, 12) * 8;
    sortedNodes.forEach((node, index) => {
      const angle = (Math.PI * 2 * index) / Math.max(sortedNodes.length, 1) - Math.PI / 2;
      const orbit = innerRadius + Math.floor(index / 6) * 18;
      positions[node.id] = {
        x: groupCenterX + Math.cos(angle) * orbit,
        y: groupCenterY + Math.sin(angle) * orbit,
      };
    });
  });

  return positions;
}

export const TYPE_STYLE = {
  DOMAIN: { color: "#8fb7ff", accent: "#3158c9", shape: "concept" },
  CONCEPT: { color: "#6ee7ff", accent: "#1d4ed8", shape: "concept" },
  PATTERN: { color: "#f59e0b", accent: "#b45309", shape: "thought" },
  EXAMPLE: { color: "#f97316", accent: "#c2410c", shape: "thought" },
  PERSON: { color: "#c084fc", accent: "#7c3aed", shape: "character" },
  TRAIT: { color: "#34d399", accent: "#0f766e", shape: "term" },
  TRIGGER: { color: "#fb7185", accent: "#be123c", shape: "trigger" },
  SIGNAL: { color: "#fda4af", accent: "#be185d", shape: "term" },
  AGENT: { color: "#fde68a", accent: "#ca8a04", shape: "episode" },
  THOUGHT: { color: "#f59e0b", accent: "#b45309", shape: "thought" },
  TERM: { color: "#34d399", accent: "#0f766e", shape: "term" },
  CHARACTER: { color: "#c084fc", accent: "#7c3aed", shape: "character" },
};

export function styleForType(type) {
  return TYPE_STYLE[String(type || "").toUpperCase()] || { color: "#94a3b8", accent: "#334155", shape: "generic" };
}

export function filterGraphPayload({
  nodes = [],
  edges = [],
  searchTerm = "",
  nodeTypes = [],
  edgeType = "",
  minWeight = 0,
  focusNodeId = "",
  neighborhoodDepth = 0,
}) {
  const allowedNodeTypes = new Set((nodeTypes || []).filter(Boolean));
  const loweredSearch = String(searchTerm || "").trim().toLowerCase();
  const searchHits = new Set();
  for (const node of nodes) {
    const haystack = [getNodeLabel(node), getNodeGloss(node), getNodeExplanation(node), ...getNodeTags(node)].join(" ").toLowerCase();
    if (loweredSearch && haystack.includes(loweredSearch)) {
      searchHits.add(String(node.id));
    }
  }
  const neighborhoodIds = focusNodeId && neighborhoodDepth > 0 ? bfsNeighborhood(focusNodeId, edges, neighborhoodDepth) : null;
  const filteredEdges = edges.filter((edge) => {
    const typeOk = !edgeType || edgeTypeOf(edge) === edgeType;
    const weightOk = edgeWeightOf(edge) >= minWeight;
    return typeOk && weightOk;
  });
  const connectedIds = new Set();
  for (const edge of filteredEdges) {
    connectedIds.add(String(edge.src_id || edge.from || ""));
    connectedIds.add(String(edge.dst_id || edge.to || ""));
  }
  const filteredNodes = nodes.filter((node) => {
    const nodeId = String(node.id);
    const typeOk = !allowedNodeTypes.size || allowedNodeTypes.has(String(node.type || "generic"));
    const searchOk = !loweredSearch || searchHits.has(nodeId);
    const neighborhoodOk = !neighborhoodIds || neighborhoodIds.has(nodeId);
    const connectedOk = connectedIds.size === 0 || connectedIds.has(nodeId) || searchHits.has(nodeId) || nodeId === String(focusNodeId || "");
    return typeOk && searchOk && neighborhoodOk && connectedOk;
  });
  const filteredNodeIds = new Set(filteredNodes.map((node) => String(node.id)));
  const visibleEdges = filteredEdges.filter((edge) => filteredNodeIds.has(String(edge.src_id || edge.from || "")) && filteredNodeIds.has(String(edge.dst_id || edge.to || "")));
  return {
    nodes: filteredNodes,
    edges: visibleEdges,
    searchHits,
    neighborhoodIds: neighborhoodIds || new Set(),
  };
}

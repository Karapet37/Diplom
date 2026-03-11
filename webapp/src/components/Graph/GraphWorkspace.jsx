import React, { useMemo, useRef, useState } from "react";
import { deterministicLayout, edgeTypeOf, edgeWeightOf, getNodeGloss, getNodeLabel, styleForType } from "../../lib/graphView";

function nodeRadius(node, degreeMap) {
  const degree = Number(degreeMap.get(String(node.id)) || 0);
  return Math.min(34, 16 + degree * 1.7);
}

function renderNodeShape(shape, x, y, radius) {
  if (shape === "thought") {
    return <rect x={x - radius} y={y - radius * 0.75} width={radius * 2} height={radius * 1.5} rx={radius * 0.5} />;
  }
  if (shape === "trigger") {
    return <polygon points={`${x},${y - radius} ${x + radius},${y} ${x},${y + radius} ${x - radius},${y}`} />;
  }
  if (shape === "term") {
    return <rect x={x - radius * 0.82} y={y - radius * 0.82} width={radius * 1.64} height={radius * 1.64} rx={radius * 0.32} />;
  }
  if (shape === "character") {
    return <circle cx={x} cy={y} r={radius} />;
  }
  if (shape === "episode") {
    return <polygon points={`${x},${y - radius} ${x + radius * 0.92},${y - radius * 0.2} ${x + radius * 0.58},${y + radius} ${x - radius * 0.58},${y + radius} ${x - radius * 0.92},${y - radius * 0.2}`} />;
  }
  return <circle cx={x} cy={y} r={radius} />;
}

export function GraphWorkspace({
  nodes,
  edges,
  loops,
  selectedNodeId,
  selectedEdgeKey,
  highlightedNodeIds,
  highlightedEdgeKeys,
  searchHitIds,
  onSelectNode,
  onSelectEdge,
  onClearSelection,
  t,
}) {
  const svgRef = useRef(null);
  const dragRef = useRef(null);
  const [viewport, setViewport] = useState({ x: 0, y: 0, scale: 1 });
  const [hoverLabel, setHoverLabel] = useState("");
  const width = 1180;
  const height = 780;

  const buildHoverText = (node) => {
    const label = getNodeLabel(node);
    const gloss = String(getNodeGloss(node) || "").trim();
    if (!gloss) {
      return label;
    }
    return `${label} - ${gloss.slice(0, 120)}`;
  };

  const degreeMap = useMemo(() => {
    const map = new Map();
    for (const node of nodes) {
      map.set(String(node.id), 0);
    }
    for (const edge of edges) {
      const src = String(edge.src_id || edge.from || "");
      const dst = String(edge.dst_id || edge.to || "");
      if (map.has(src)) {
        map.set(src, Number(map.get(src) || 0) + 1);
      }
      if (map.has(dst)) {
        map.set(dst, Number(map.get(dst) || 0) + 1);
      }
    }
    return map;
  }, [nodes, edges]);

  const positions = useMemo(() => deterministicLayout(nodes, edges, width, height), [nodes, edges]);

  const onWheel = (event) => {
    event.preventDefault();
    const delta = event.deltaY > 0 ? -0.08 : 0.08;
    setViewport((current) => ({
      ...current,
      scale: Math.min(2.4, Math.max(0.55, Number((current.scale + delta).toFixed(3)))),
    }));
  };

  const onPointerDown = (event) => {
    if (event.target.dataset.kind === "node" || event.target.dataset.kind === "edge") {
      return;
    }
    dragRef.current = { x: event.clientX, y: event.clientY, viewport };
  };

  const onPointerMove = (event) => {
    if (!dragRef.current) {
      return;
    }
    const diffX = event.clientX - dragRef.current.x;
    const diffY = event.clientY - dragRef.current.y;
    setViewport({
      ...dragRef.current.viewport,
      x: dragRef.current.viewport.x + diffX,
      y: dragRef.current.viewport.y + diffY,
    });
  };

  const onPointerUp = () => {
    dragRef.current = null;
  };

  const { loopNodeIds, twoCycleKeys } = loops;
  const highlightedNodes = highlightedNodeIds || new Set();
  const highlightedEdges = highlightedEdgeKeys || new Set();
  const searchHits = searchHitIds || new Set();

  return (
    <section className="graph-workspace glass-panel">
      <header className="panel-heading compact">
        <div>
          <p className="eyebrow">{t("graph_body")}</p>
          <h2>{t("graph_workspace")}</h2>
        </div>
        <div className="graph-legend">
          <span>{t("graph_drag")}</span>
          <span>{t("graph_zoom")}</span>
          {hoverLabel ? <strong>{hoverLabel}</strong> : null}
        </div>
      </header>
      <div className="graph-surface" onWheel={onWheel} onPointerMove={onPointerMove} onPointerUp={onPointerUp} onPointerLeave={onPointerUp}>
        {!nodes.length ? (
          <div className="empty-state large">
            <h3>{t("graph_empty_title")}</h3>
            <p>{t("graph_empty_text")}</p>
          </div>
        ) : (
          <svg
            ref={svgRef}
            viewBox={`0 0 ${width} ${height}`}
            className="graph-svg"
            onPointerDown={onPointerDown}
            onDoubleClick={onClearSelection}
          >
            <defs>
              <filter id="graphGlow" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="6" result="blur" />
                <feMerge>
                  <feMergeNode in="blur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            </defs>
            <rect x="0" y="0" width={width} height={height} className="graph-background" />
            <g transform={`translate(${viewport.x} ${viewport.y}) scale(${viewport.scale})`}>
              {edges.map((edge) => {
                const key = edge.edge_key || `${edge.src_id || edge.from}|${edgeTypeOf(edge)}|${edge.dst_id || edge.to}`;
                const from = positions[String(edge.src_id || edge.from)];
                const to = positions[String(edge.dst_id || edge.to)];
                if (!from || !to) {
                  return null;
                }
                const isSelfLoop = String(edge.src_id || edge.from) === String(edge.dst_id || edge.to);
                const isSelected = selectedEdgeKey === key;
                const inPath = highlightedEdges.has(key);
                const inTwoCycle = twoCycleKeys.has(`${edge.src_id || edge.from}|${edge.dst_id || edge.to}`);
                const weight = edgeWeightOf(edge);
                const strokeWidth = Math.max(1.1, Math.min(4.6, 1 + weight * 2.2));
                if (isSelfLoop) {
                  const radius = nodeRadius({ id: edge.src_id || edge.from }, degreeMap) + 10;
                  return (
                    <g key={key}>
                      <circle
                        data-kind="edge"
                        className={`graph-edge self-loop ${isSelected ? "selected" : ""} ${inPath ? "path" : ""}`}
                        cx={from.x}
                        cy={from.y}
                        r={radius}
                        strokeWidth={strokeWidth}
                        onClick={() => onSelectEdge(edge)}
                      />
                    </g>
                  );
                }
                return (
                  <g key={key}>
                    <line
                      data-kind="edge"
                      x1={from.x}
                      y1={from.y}
                      x2={to.x}
                      y2={to.y}
                      className={`graph-edge ${isSelected ? "selected" : ""} ${inPath ? "path" : ""} ${inTwoCycle ? "loop" : ""}`}
                      strokeWidth={strokeWidth}
                      onClick={() => onSelectEdge(edge)}
                    />
                    <text
                      x={(from.x + to.x) / 2}
                      y={(from.y + to.y) / 2 - 6}
                      className="edge-label"
                    >
                      {edgeTypeOf(edge)}
                    </text>
                  </g>
                );
              })}
              {nodes.map((node) => {
                const pos = positions[String(node.id)];
                if (!pos) {
                  return null;
                }
                const style = styleForType(node.type);
                const radius = nodeRadius(node, degreeMap);
                const isSelected = String(selectedNodeId || "") === String(node.id);
                const isSearchHit = searchHits.has(String(node.id));
                const inPath = highlightedNodes.has(String(node.id));
                const looped = loopNodeIds.has(String(node.id));
                return (
                  <g
                    key={node.id}
                    transform={`translate(${pos.x} ${pos.y})`}
                    onMouseEnter={() => setHoverLabel(buildHoverText(node))}
                    onMouseLeave={() => setHoverLabel("")}
                  >
                    {looped ? <circle className="graph-node-loop" cx="0" cy="0" r={radius + 8} /> : null}
                    <g
                      data-kind="node"
                      className={`graph-node ${isSelected ? "selected" : ""} ${isSearchHit ? "search-hit" : ""} ${inPath ? "path" : ""}`}
                      fill={style.color}
                      stroke={style.accent}
                      filter={isSelected || inPath ? "url(#graphGlow)" : undefined}
                      onClick={() => onSelectNode(node)}
                    >
                      {renderNodeShape(style.shape, 0, 0, radius)}
                    </g>
                    <text x="0" y={radius + 18} className="graph-node-label">
                      {getNodeLabel(node)}
                    </text>
                    <text x="0" y={4} className="graph-node-type">
                      {String(node.type || "").toUpperCase()}
                    </text>
                  </g>
                );
              })}
            </g>
          </svg>
        )}
      </div>
    </section>
  );
}

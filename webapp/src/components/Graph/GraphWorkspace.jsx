import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  deterministicLayoutWithOptions,
  edgeTypeOf,
  edgeWeightOf,
  getNodeClusterLabel,
  getNodeGloss,
  getNodeLabel,
  getNodeLifecycleState,
  getNodeTranslationLine,
  styleForType,
} from "../../lib/graphView";

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
  onOpenBranch,
  rootNodeId,
  windowTitle,
  windowSubtitle,
  onCloseWindow,
  isBranchWindow = false,
  diagnostics = null,
  layoutSpread = 1.34,
  showEdgeLabels = true,
  t,
}) {
  const svgRef = useRef(null);
  const surfaceRef = useRef(null);
  const dragRef = useRef(null);
  const lastTouchRef = useRef({ nodeId: "", at: 0 });
  const [viewport, setViewport] = useState({ x: 0, y: 0, scale: 1 });
  const [hoverLabel, setHoverLabel] = useState("");
  const width = isBranchWindow ? 1400 : 1760;
  const height = isBranchWindow ? 920 : 1160;

  const buildHoverText = (node) => {
    const label = getNodeLabel(node);
    const translation = String(getNodeTranslationLine(node) || "").trim();
    const gloss = String(getNodeGloss(node) || "").trim();
    if (!gloss && !translation) {
      return label;
    }
    const detail = translation || gloss;
    return `${label} - ${detail.slice(0, 160)}`;
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

  const positions = useMemo(
    () => deterministicLayoutWithOptions(
      nodes,
      edges,
      width,
      height,
      { centerNodeId: rootNodeId, spread: Math.max(0.82, Number(layoutSpread || 1)) * (isBranchWindow ? 0.9 : 1) }
    ),
    [edges, height, isBranchWindow, layoutSpread, nodes, rootNodeId, width],
  );

  const onWheel = (event) => {
    event.preventDefault();
    event.stopPropagation();
    const delta = event.deltaY > 0 ? -0.08 : 0.08;
    setViewport((current) => ({
      ...current,
      scale: Math.min(2.4, Math.max(0.55, Number((current.scale + delta).toFixed(3)))),
    }));
  };

  useEffect(() => {
    const surface = surfaceRef.current;
    if (!surface) {
      return undefined;
    }
    const handleWheel = (event) => {
      onWheel(event);
    };
    surface.addEventListener("wheel", handleWheel, { passive: false });
    return () => {
      surface.removeEventListener("wheel", handleWheel);
    };
  }, []);

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

  const handleBranchOpen = (node, event) => {
    if (!onOpenBranch || !node) {
      return;
    }
    if (event) {
      event.stopPropagation();
    }
    onOpenBranch(node, rootNodeId || "");
  };

  const handleNodePointerUp = (node, event) => {
    if (!onOpenBranch || event.pointerType !== "touch") {
      return;
    }
    const now = Date.now();
    const nodeId = String(node?.id || "");
    if (lastTouchRef.current.nodeId === nodeId && now - lastTouchRef.current.at <= 320) {
      handleBranchOpen(node, event);
      lastTouchRef.current = { nodeId: "", at: 0 };
      return;
    }
    lastTouchRef.current = { nodeId, at: now };
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
          <h2>{windowTitle || t("graph_workspace")}</h2>
          {windowSubtitle ? <span className="graph-window-subtitle">{windowSubtitle}</span> : null}
        </div>
        <div className="graph-legend">
          <button type="button" className="button-secondary graph-mini-button" onClick={() => setViewport((current) => ({ ...current, scale: Math.max(0.55, Number((current.scale - 0.12).toFixed(3))) }))}>-</button>
          <button type="button" className="button-secondary graph-mini-button" onClick={() => setViewport({ x: 0, y: 0, scale: 1 })}>{t("graph_reset_view")}</button>
          <button type="button" className="button-secondary graph-mini-button" onClick={() => setViewport((current) => ({ ...current, scale: Math.min(2.4, Number((current.scale + 0.12).toFixed(3))) }))}>+</button>
          <span>{t("graph_drag")}</span>
          <span>{t("graph_zoom")}</span>
          <span>{`${nodes.length} n / ${edges.length} e`}</span>
          {diagnostics ? (
            <span>
              {(diagnostics.active_node_count || 0)}
              {" active / "}
              {(diagnostics.weak_node_count || 0)}
              {" weak / "}
              {(diagnostics.suspect_node_count || 0)}
              {" suspect"}
            </span>
          ) : null}
          {isBranchWindow && onCloseWindow ? <button type="button" className="button-secondary" onClick={onCloseWindow}>×</button> : null}
          {hoverLabel ? <strong>{hoverLabel}</strong> : null}
        </div>
      </header>
      <div ref={surfaceRef} className="graph-surface" onPointerMove={onPointerMove} onPointerUp={onPointerUp} onPointerLeave={onPointerUp}>
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
                const edgeId = edge.id || key;
                const from = positions[String(edge.src_id || edge.from)];
                const to = positions[String(edge.dst_id || edge.to)];
                if (!from || !to) {
                  return null;
                }
                const isSelfLoop = String(edge.src_id || edge.from) === String(edge.dst_id || edge.to);
                const isSelected = selectedEdgeKey === key || selectedEdgeKey === edgeId;
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
                    {showEdgeLabels ? (
                      <text
                        x={(from.x + to.x) / 2}
                        y={(from.y + to.y) / 2 - 6}
                        className="edge-label"
                      >
                        {edgeTypeOf(edge)}
                      </text>
                    ) : null}
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
                const isRoot = rootNodeId && String(rootNodeId) === String(node.id);
                const translationLine = String(getNodeTranslationLine(node) || "").trim();
                const lifecycleState = getNodeLifecycleState(node);
                const clusterLabel = getNodeClusterLabel(node);
                return (
                  <g
                    key={node.id}
                    transform={`translate(${pos.x} ${pos.y})`}
                    onMouseEnter={() => setHoverLabel(buildHoverText(node))}
                    onMouseLeave={() => setHoverLabel("")}
                    onDoubleClick={(event) => handleBranchOpen(node, event)}
                    onPointerUp={(event) => handleNodePointerUp(node, event)}
                  >
                    {looped || isRoot ? <circle className={`graph-node-loop ${isRoot ? "root" : ""}`} cx="0" cy="0" r={radius + (isRoot ? 12 : 8)} /> : null}
                    <g
                      data-kind="node"
                      className={`graph-node ${isSelected ? "selected" : ""} ${isSearchHit ? "search-hit" : ""} ${inPath ? "path" : ""} ${isRoot ? "root" : ""} lifecycle-${lifecycleState}`}
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
                    {translationLine ? (
                      <text x="0" y={radius + 31} className="graph-node-translation">
                        {translationLine}
                      </text>
                    ) : null}
                    <text x="0" y={4} className="graph-node-type">
                      {String(node.type || "").toUpperCase()}
                    </text>
                    <text x="0" y={18} className={`graph-node-state graph-node-state-${lifecycleState}`}>
                      {lifecycleState.toUpperCase()}
                    </text>
                    {clusterLabel ? (
                      <text x="0" y={radius + 44} className="graph-node-cluster">
                        {clusterLabel}
                      </text>
                    ) : null}
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

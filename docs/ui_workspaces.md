# UI Workspaces

The active frontend is split into two workspaces.

## Chat Workspace

Layout:
- left sidebar: sessions and chat tools
- right main panel: scrollable message thread
- fixed input bar at the bottom

Rules:
- messages scroll inside the thread area
- the composer stays at the bottom
- no extra tools are placed between the thread and the input bar

Session storage:
- `data/sessions/{session_id}.json`

## Graph Workspace

Layout:
- left sidebar: sessions, graph query, quick foundation loading
- graph canvas in the main area
- node editor panel on the right

The graph workspace renders only a relevant subgraph for the current query.

Subgraph flow:
- current session query or manual graph query
- `/api/cognitive/graph/subgraph`
- seed matches + neighbor expansion
- filtered nodes and edges rendered in the canvas

## Graph Zones

Graph material remains split into:
- `graph/verified/`
- `graph/pending/`

Chat sessions are stored separately in:
- `data/sessions/`

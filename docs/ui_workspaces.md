# UI Workspaces

The frontend is reduced to two workspaces only.

## Chat

Purpose:

- user conversation
- session switching
- personality selection
- fixed composer

Rules:

- render only user and assistant messages
- keep the composer fixed at the bottom
- do not render graph jobs or internal pipeline logs in the thread

## Graph

Purpose:

- inspect graph nodes and edges
- rebuild graph from session files
- inspect personalities

Rules:

- the graph should answer three questions for every node:
  1. who / what is this node
  2. what is it like
  3. how does it act through relations
- the graph view should use `memory/graphs/nodes.json` and `memory/graphs/edges.json` as the source of truth
- do not expose debug-only controller panels in the MVP UI

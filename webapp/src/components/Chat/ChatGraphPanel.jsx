import React, { useEffect, useMemo, useRef } from "react";

function DiffNodeCard({ item }) {
  return (
    <article className="branch-card">
      <strong>{item.label || `${item.type}:${item.id}`}</strong>
      <span>{item.type}</span>
      {item.short_gloss ? <p>{item.short_gloss}</p> : null}
    </article>
  );
}

function DiffEdgeCard({ item }) {
  return (
    <article className="branch-card">
      <strong>{item.src_label} → {item.dst_label}</strong>
      <span>{item.type} · {Number(item.weight || 0).toFixed(2)}</span>
    </article>
  );
}

function ThreadMessage({ item, t }) {
  const role = item.role === "user" ? "user" : "assistant";
  const metadata = item.metadata || {};
  const diff = metadata.graphDiff || {};
  const job = metadata.graphJob || null;
  const contextNodes = metadata.contextNodes || [];
  const hasDiff = diff.attached && (((diff.nodes || []).length > 0) || ((diff.edges || []).length > 0));

  return (
    <article className={`chat-thread-item ${role}`}>
      <header>
        <strong>{role === "user" ? t("chat_you") : t("chat_assistant")}</strong>
        <span>{item.timestamp}</span>
      </header>
      <p>{item.message}</p>

      {hasDiff ? (
        <details className="chat-message-details" open>
          <summary>{t("chat_graph_diff")}</summary>
          <div className="summary-strip compact">
            <article className="summary-mini-card"><span>{t("chat_new_nodes")}</span><strong>{diff.node_count || 0}</strong></article>
            <article className="summary-mini-card"><span>{t("chat_new_edges")}</span><strong>{diff.edge_count || 0}</strong></article>
          </div>
          <div className="controller-grid chat-details-grid">
            {(diff.nodes || []).length ? (
              <section className="controller-card wide">
                <h3>{t("chat_new_nodes")}</h3>
                <div className="branch-list">
                  {(diff.nodes || []).map((node) => <DiffNodeCard key={node.id} item={node} />)}
                </div>
              </section>
            ) : null}
            {(diff.edges || []).length ? (
              <section className="controller-card wide">
                <h3>{t("chat_new_edges")}</h3>
                <div className="branch-list">
                  {(diff.edges || []).map((edge, index) => (
                    <DiffEdgeCard key={`${edge.src_id}-${edge.type}-${edge.dst_id}-${index}`} item={edge} />
                  ))}
                </div>
              </section>
            ) : null}
          </div>
        </details>
      ) : null}

      {job ? (
        <details className="chat-message-details" open>
          <summary>{t("chat_graph_build_status")}</summary>
          <div className="controller-grid chat-details-grid">
            <section className="controller-card wide">
              <div className="flag-row"><span>{t("chat_graph_job_state")}</span><strong>{job.status || "unknown"}</strong></div>
              {job.reason ? <div className="flag-row"><span>{t("chat_graph_job_reason")}</span><strong>{job.reason}</strong></div> : null}
              {(job.requests || []).length ? (
                <div className="stack-gap-sm">
                  <h3>{t("chat_graph_job_requests")}</h3>
                  <ul className="dense-list compact">
                    {job.requests.map((request) => <li key={request}>{request}</li>)}
                  </ul>
                </div>
              ) : null}
            </section>
          </div>
        </details>
      ) : null}

      {contextNodes.length ? (
        <details className="chat-message-details">
          <summary>{t("chat_context_nodes")}</summary>
          <div className="branch-list">
            {contextNodes.slice(0, 8).map((node) => (
              <article key={node.node_id} className="branch-card">
                <strong>{node.name}</strong>
                <span>{node.type} · {Number(node.score || 0).toFixed(2)}</span>
                <p>{node.description}</p>
              </article>
            ))}
          </div>
        </details>
      ) : null}
    </article>
  );
}

export function ChatGraphPanel({
  session,
  value,
  onChange,
  running,
  progress,
  onRun,
  t,
}) {
  const threadRef = useRef(null);
  const messages = useMemo(() => listMessages(session), [session]);

  useEffect(() => {
    const node = threadRef.current;
    if (!node) return;
    node.scrollTop = node.scrollHeight;
  }, [messages.length, running]);

  const onSubmit = (event) => {
    event.preventDefault();
    if (!value.trim() || running) return;
    onRun();
  };

  return (
    <section className="chat-workspace-panel glass-panel">
      <header className="panel-heading compact">
        <div>
          <p className="eyebrow">{t("chat_eyebrow")}</p>
          <h2>{session?.title || t("chat_title")}</h2>
        </div>
        {progress ? <span className="chat-progress-label">{progress}</span> : null}
      </header>

      <div ref={threadRef} className="chat-thread-scroll">
        {!messages.length ? (
          <div className="empty-state large">
            <h3>{t("chat_empty_title")}</h3>
            <p>{t("chat_empty_text")}</p>
          </div>
        ) : (
          <div className="chat-thread">
            {messages.map((item) => <ThreadMessage key={item.id} item={item} t={t} />)}
          </div>
        )}
      </div>

      <form className="chat-composer-fixed" onSubmit={onSubmit}>
        <label className="field-stack">
          <span>{t("chat_message")}</span>
          <textarea
            value={value}
            onChange={(event) => onChange(event.target.value)}
            placeholder={t("chat_message_placeholder")}
          />
        </label>
        <div className="chat-composer-actions">
          <button type="submit" disabled={running || !value.trim()}>
            {running ? t("chat_running") : t("chat_send")}
          </button>
        </div>
      </form>
    </section>
  );
}

function listMessages(session) {
  if (!session || !Array.isArray(session.messages)) {
    return [];
  }
  return session.messages.map((item, index) => ({
    id: item.id || `${item.role || "msg"}-${index}`,
    role: item.role || "assistant",
    message: item.message || "",
    timestamp: item.timestamp || "",
    metadata: item.metadata || {},
  }));
}

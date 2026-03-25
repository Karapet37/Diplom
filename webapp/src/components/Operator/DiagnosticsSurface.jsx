import React from 'react';

function metricRows(metrics) {
  if (!metrics || typeof metrics !== 'object') return [];
  return Object.entries(metrics);
}

export function DiagnosticsSurface({
  diagnostics,
  loading,
  onRefresh,
  activeTrace,
  onSelectTrace,
  t,
}) {
  const metrics = diagnostics?.metrics || {};
  const traces = diagnostics?.traces || [];
  const graphHealth = diagnostics?.graphHealth || {};
  const counters = metrics?.counters || {};
  const rates = metrics?.rates || {};
  const stageTimings = metrics?.stage_timings_ms || {};

  return (
    <div className="operator-surface-grid diagnostics-surface-grid">
      <section className="workspace-panel glass-panel">
        <header className="panel-heading compact">
          <div>
            <p className="eyebrow">{t('diagnostics_title')}</p>
            <h2>{t('diagnostics_metrics')}</h2>
          </div>
          <button type="button" onClick={onRefresh} disabled={loading}>
            {loading ? t('top_refreshing') : t('top_refresh')}
          </button>
        </header>
        <div className="operator-stack">
          <div className="operator-kv-grid">
            <div><span>fallback rate</span><strong>{Number(rates.fallback_rate ?? 0).toFixed(4)}</strong></div>
            <div><span>recent traces</span><strong>{metrics?.recent_trace_count ?? 0}</strong></div>
            <div><span>context avg</span><strong>{metrics?.context_tokens?.avg ?? 0}</strong></div>
            <div><span>context max</span><strong>{metrics?.context_tokens?.max ?? 0}</strong></div>
          </div>
          <section className="operator-card-block">
            <h3>{t('diagnostics_stage_timings')}</h3>
            <div className="operator-table-wrap">
              <table className="operator-table">
                <thead>
                  <tr>
                    <th>Stage</th>
                    <th>Count</th>
                    <th>Avg ms</th>
                    <th>Max ms</th>
                  </tr>
                </thead>
                <tbody>
                  {metricRows(stageTimings).map(([name, item]) => (
                    <tr key={name}>
                      <td>{name}</td>
                      <td>{item.count}</td>
                      <td>{Number(item.avg_ms ?? 0).toFixed(2)}</td>
                      <td>{Number(item.max_ms ?? 0).toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
          <section className="operator-card-block">
            <h3>{t('diagnostics_counters')}</h3>
            <div className="operator-table-wrap">
              <table className="operator-table">
                <thead>
                  <tr>
                    <th>Counter</th>
                    <th>Value</th>
                  </tr>
                </thead>
                <tbody>
                  {metricRows(counters).slice(0, 18).map(([name, value]) => (
                    <tr key={name}>
                      <td>{name}</td>
                      <td>{String(value)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </div>
      </section>

      <section className="workspace-panel glass-panel operator-inspector-panel">
        <header className="panel-heading compact">
          <div>
            <p className="eyebrow">{t('diagnostics_graph_health')}</p>
            <h2>{t('diagnostics_trace_inspector')}</h2>
          </div>
        </header>
        <div className="operator-stack">
          <section className="operator-card-block">
            <h3>{t('diagnostics_graph_health')}</h3>
            <div className="operator-kv-grid">
              <div><span>nodes</span><strong>{graphHealth.node_count ?? 0}</strong></div>
              <div><span>edges</span><strong>{graphHealth.edge_count ?? 0}</strong></div>
              <div><span>duplicate rate</span><strong>{graphHealth.duplicate_rate ?? 0}</strong></div>
              <div><span>orphan rate</span><strong>{graphHealth.orphan_rate ?? 0}</strong></div>
              <div><span>suspect</span><strong>{graphHealth.suspect_node_count ?? 0}</strong></div>
              <div><span>archived</span><strong>{graphHealth.archived_node_count ?? 0}</strong></div>
            </div>
          </section>

          <section className="operator-card-block">
            <h3>{t('diagnostics_recent_traces')}</h3>
            {traces.length ? (
              <ul className="dense-list clickable-list">
                {traces.map((trace) => (
                  <li key={trace.request_id}>
                    <button type="button" className={`trace-row-button ${activeTrace?.request_id === trace.request_id ? 'active' : ''}`} onClick={() => onSelectTrace(trace)}>
                      <strong>{trace.request_id.slice(0, 10)}</strong>
                      {' '}
                      {trace.persona_name || 'no-persona'}
                      {' '}
                      <span>{Number(trace.total_ms ?? 0).toFixed(2)} ms</span>
                    </button>
                  </li>
                ))}
              </ul>
            ) : <p className="empty-inline">No recent traces.</p>}
          </section>

          <section className="operator-card-block">
            <h3>{t('chat_trace_inspection')}</h3>
            {activeTrace ? (
              <>
                <div className="operator-kv-grid">
                  <div><span>request</span><strong>{activeTrace.request_id}</strong></div>
                  <div><span>route</span><strong>{activeTrace.route}</strong></div>
                  <div><span>status</span><strong>{activeTrace.status}</strong></div>
                  <div><span>total ms</span><strong>{Number(activeTrace.total_ms ?? 0).toFixed(2)}</strong></div>
                </div>
                <pre className="operator-pre">{JSON.stringify(activeTrace, null, 2)}</pre>
              </>
            ) : <p className="empty-inline">Select a trace.</p>}
          </section>
        </div>
      </section>
    </div>
  );
}

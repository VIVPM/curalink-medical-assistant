export default function PipelinePanel({ meta, collapsed, onToggle }) {
  if (!meta) return null;

  const timings = meta.stage_timings_ms || {};
  const counts = meta.retrieval_counts || {};
  const citations = meta.citation_stats || {};
  const warnings = meta.warnings || [];
  const totalMs = timings.total || Object.values(timings).reduce((a, b) => a + b, 0);

  const STAGES = [
    { key: "query_expansion", label: "Query Expansion" },
    { key: "retrieval", label: "Retrieval" },
    { key: "normalization", label: "Normalization" },
    { key: "ranking", label: "Ranking" },
    { key: "context_build", label: "Context Build" },
    { key: "llm", label: "LLM Reasoning" },
    { key: "assembly", label: "Assembly" },
  ];

  const maxTiming = Math.max(
    ...Object.values(timings).filter((v) => v !== totalMs),
    1
  );

  return (
    <aside className={`pipeline-panel ${collapsed ? "pipeline-panel-collapsed" : ""}`}>
      <div className="pipeline-panel-header">
        <button
          className="pipeline-toggle"
          onClick={onToggle}
          title={collapsed ? "Expand diagnostics" : "Collapse diagnostics"}
        >
          {collapsed ? "\u2039" : "\u203A"}
        </button>
        {!collapsed && <h3>Diagnostics</h3>}
        {!collapsed && (
          <span className="pipeline-total-time">
            {(totalMs / 1000).toFixed(1)}s total
          </span>
        )}
      </div>

      {!collapsed && (
        <>
          {/* Stage Timings */}
          <div className="pipeline-section">
            <h4>Stage Timings</h4>
            <div className="timing-list">
              {STAGES.map(({ key, label }) => {
                const ms = timings[key];
                if (ms == null) return null;
                return (
                  <div key={key} className="timing-row">
                    <div className="timing-label">{label}</div>
                    <div className="timing-bar-track">
                      <div
                        className="timing-bar-fill"
                        style={{
                          width: `${Math.max((ms / maxTiming) * 100, 2)}%`,
                        }}
                      />
                    </div>
                    <div className="timing-value">
                      {ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Retrieval Counts */}
          <div className="pipeline-section">
            <h4>Retrieval Counts</h4>
            <div className="count-grid">
              {counts.pubmed != null && (
                <div className="count-item">
                  <span className="count-number">{counts.pubmed}</span>
                  <span className="count-label">PubMed</span>
                </div>
              )}
              {counts.openalex != null && (
                <div className="count-item">
                  <span className="count-number">{counts.openalex}</span>
                  <span className="count-label">OpenAlex</span>
                </div>
              )}
              {counts.trials != null && (
                <div className="count-item">
                  <span className="count-number">{counts.trials}</span>
                  <span className="count-label">Trials</span>
                </div>
              )}
              {counts.pinecone != null && (
                <div className="count-item">
                  <span className="count-number">{counts.pinecone}</span>
                  <span className="count-label">Pinecone</span>
                </div>
              )}
            </div>

            <div className="retrieval-funnel">
              {counts.after_dedupe != null && (
                <div className="funnel-step">
                  <span className="funnel-arrow">&#8595;</span>
                  <span className="funnel-value">{counts.after_dedupe}</span>
                  <span className="funnel-label">after deduplication</span>
                </div>
              )}
              {counts.after_filter != null && (
                <div className="funnel-step">
                  <span className="funnel-arrow">&#8595;</span>
                  <span className="funnel-value">{counts.after_filter}</span>
                  <span className="funnel-label">after filter</span>
                </div>
              )}
              {counts.after_ranking != null && (
                <div className="funnel-step">
                  <span className="funnel-arrow">&#8595;</span>
                  <span className="funnel-value">{counts.after_ranking}</span>
                  <span className="funnel-label">final docs</span>
                </div>
              )}
            </div>
          </div>

          {/* Citation Stats */}
          <div className="pipeline-section">
            <h4>Citations</h4>
            <div className="citation-stats">
              <div className="citation-row">
                <span className="citation-label">Total</span>
                <span className="citation-value">{citations.total ?? 0}</span>
              </div>
              <div className="citation-row verified">
                <span className="citation-label">Verified</span>
                <span className="citation-value">{citations.verified ?? 0}</span>
              </div>
              {(citations.unverified ?? 0) > 0 && (
                <div className="citation-row unverified">
                  <span className="citation-label">Unverified</span>
                  <span className="citation-value">{citations.unverified}</span>
                </div>
              )}
            </div>
          </div>

          {/* Warnings */}
          {warnings.length > 0 && (
            <div className="pipeline-section pipeline-warnings">
              <h4>Warnings</h4>
              <ul>
                {warnings.map((w, i) => (
                  <li key={i}>{w}</li>
                ))}
              </ul>
            </div>
          )}
        </>
      )}
    </aside>
  );
}

import InsightCard from "./InsightCard";
import TrialCard from "./TrialCard";

export default function StructuredResponse({ data, onFollowUp }) {
  if (!data) return null;

  const { overview, insights, trials, recommendations, abstain_reason, pipelineMeta, follow_up_questions } = data;
  const warnings = pipelineMeta?.warnings || [];
  const hasSourceWarnings = warnings.some(
    (w) => w.includes("_failed") || w.includes("_timeout")
  );

  // Abstain state
  if (abstain_reason) {
    return (
      <div className="structured-response abstain">
        <div className="overview">{overview}</div>
        <div className="abstain-reason">
          <span className="abstain-icon">i</span>
          <div>
            <p>{abstain_reason}</p>
            {data.suggestion && <p className="suggestion">{data.suggestion}</p>}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="structured-response">
      {/* Source warnings banner */}
      {hasSourceWarnings && (
        <div className="warning-banner">
          <span className="warning-icon">!</span>
          <p>
            Some data sources had issues. Results may be incomplete.
            {warnings
              .filter((w) => w.includes("_failed"))
              .map((w) => {
                const src = w.split("_failed")[0];
                return ` ${src} unavailable.`;
              })}
          </p>
        </div>
      )}

      {overview && (
        <div className="section overview-section">
          <h3>Overview</h3>
          <p>{overview}</p>
        </div>
      )}

      {insights && insights.length > 0 && (
        <div className="section insights-section">
          <h3>Research Insights</h3>
          {insights.map((ins, i) => (
            <InsightCard key={i} insight={ins} index={i} />
          ))}
        </div>
      )}

      {/* Empty insights state */}
      {(!insights || insights.length === 0) && !abstain_reason && (
        <div className="section empty-section">
          <h3>Research Insights</h3>
          <p className="empty-msg">No relevant publications found for this query.</p>
        </div>
      )}

      {trials && trials.length > 0 && (
        <div className="section trials-section">
          <h3>Clinical Trials</h3>
          {trials.map((trial, i) => (
            <TrialCard key={i} trial={trial} />
          ))}
        </div>
      )}

      {/* Empty trials state */}
      {(!trials || trials.length === 0) && !abstain_reason && (
        <div className="section empty-section">
          <h3>Clinical Trials</h3>
          <p className="empty-msg">No matching clinical trials found.</p>
        </div>
      )}

      {recommendations && recommendations.length > 0 && (
        <div className="section recommendations-section">
          <h3>Personalized Recommendations</h3>
          <ul className="recommendations-list">
            {recommendations.map((rec, i) => (
              <li key={i} className="recommendation-item">{rec}</li>
            ))}
          </ul>
        </div>
      )}

      {follow_up_questions && follow_up_questions.length > 0 && onFollowUp && (
        <div className="section follow-up-section">
          <h3>Follow-up Questions</h3>
          <div className="follow-up-chips">
            {follow_up_questions.map((q, i) => (
              <button key={i} className="follow-up-chip" onClick={() => onFollowUp(q)}>
                {q}
              </button>
            ))}
          </div>
        </div>
      )}

    </div>
  );
}

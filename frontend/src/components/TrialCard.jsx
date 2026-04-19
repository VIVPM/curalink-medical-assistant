export default function TrialCard({ trial }) {
  return (
    <div className="trial-card">
      <div className="trial-header">
        <span className={`trial-status status-${(trial.status || "unknown").toLowerCase()}`}>
          {trial.status || "Unknown"}
        </span>
        {trial.nct_id && <span className="trial-nct">{trial.nct_id}</span>}
      </div>
      <h4 className="trial-title">{trial.title}</h4>
      {trial.relevance && <p className="trial-relevance">{trial.relevance}</p>}
      {trial.age_range && (
        <div className="trial-section">
          <span className="trial-label">Age:</span>
          <p>{trial.age_range}</p>
        </div>
      )}
      {trial.eligibility_summary && (
        <div className="trial-section">
          <span className="trial-label">Eligibility:</span>
          <p>{trial.eligibility_summary.slice(0, 400)}{trial.eligibility_summary.length > 400 ? "..." : ""}</p>
        </div>
      )}
      {trial.location && (
        <div className="trial-section">
          <span className="trial-label">Location:</span>
          <p>{trial.location}</p>
        </div>
      )}
      {trial.contact && trial.contact.name && (
        <div className="trial-section">
          <span className="trial-label">Contact:</span>
          <p>
            {trial.contact.name}
            {trial.contact.email && ` · ${trial.contact.email}`}
            {trial.contact.phone && ` · ${trial.contact.phone}`}
          </p>
        </div>
      )}
      {trial.start_date && (
        <div className="trial-section">
          <span className="trial-label">Start:</span>
          <p>{trial.start_date}</p>
        </div>
      )}
      {trial.source_details && trial.source_details[0]?.url && (
        <a href={trial.source_details[0].url} target="_blank" rel="noreferrer" className="source-link">
          View on ClinicalTrials.gov
        </a>
      )}
    </div>
  );
}

const STAGES = [
  { key: "query_expansion", label: "Expanding query" },
  { key: "retrieval", label: "Fetching sources" },
  { key: "normalization", label: "Normalizing" },
  { key: "ranking", label: "Ranking docs" },
  { key: "context_build", label: "Building context" },
  { key: "llm", label: "Generating answer" },
  { key: "assembly", label: "Assembling" },
];

function getStageIndex(stage) {
  if (!stage || stage === "starting") return -1;
  return STAGES.findIndex((s) => s.key === stage);
}

export default function PipelineProgress({ stage, retrievalCounts }) {
  const activeIdx = getStageIndex(stage);
  const progressPct = Math.max(((activeIdx + 1) / STAGES.length) * 100, 3);

  // Show retrieval counts once retrieval is done (activeIdx > 1 means past retrieval)
  const showCounts = retrievalCounts && activeIdx >= 2;
  // Show retrieval as actively loading (stage is retrieval)
  const retrievalActive = activeIdx === 1;

  return (
    <div className="pipeline-progress">
      {/* Progress bar */}
      <div className="pp-bar-track">
        <div className="pp-bar-fill" style={{ width: `${progressPct}%` }} />
      </div>

      {/* Stage steps */}
      <div className="pp-stages">
        {STAGES.map((s, i) => {
          let status = "pending";
          if (i < activeIdx) status = "done";
          else if (i === activeIdx) status = "active";

          return (
            <div key={s.key}>
              <div className={`pp-stage ${status}`}>
                <div className={`pp-dot ${status}`}>
                  {status === "done" && <span>&#10003;</span>}
                  {status === "active" && <span className="pp-pulse" />}
                </div>
                <span className="pp-label">{s.label}</span>
              </div>

              {/* Retrieval counts shown under "Fetching sources" once done */}
              {s.key === "retrieval" && (showCounts || retrievalActive) && (
                <div className="pp-retrieval-counts">
                  <div className={`pp-rc-item ${retrievalCounts?.pubmed != null ? "loaded" : ""}`}>
                    <span className="pp-rc-name">PubMed</span>
                    <span className="pp-rc-value">
                      {retrievalCounts?.pubmed != null ? retrievalCounts.pubmed : (retrievalActive ? "..." : "-")}
                    </span>
                  </div>
                  <div className={`pp-rc-item ${retrievalCounts?.openalex != null ? "loaded" : ""}`}>
                    <span className="pp-rc-name">OpenAlex</span>
                    <span className="pp-rc-value">
                      {retrievalCounts?.openalex != null ? retrievalCounts.openalex : (retrievalActive ? "..." : "-")}
                    </span>
                  </div>
                  <div className={`pp-rc-item ${retrievalCounts?.trials != null ? "loaded" : ""}`}>
                    <span className="pp-rc-name">ClinicalTrials</span>
                    <span className="pp-rc-value">
                      {retrievalCounts?.trials != null ? retrievalCounts.trials : (retrievalActive ? "..." : "-")}
                    </span>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

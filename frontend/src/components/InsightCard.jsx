import { useState } from "react";

function dedupeSources(sources) {
  if (!sources || sources.length === 0) return [];

  const seen = new Map();
  for (const src of sources) {
    // Key by normalized title to catch same paper from different platforms
    const key = (src.title || "").toLowerCase().trim().slice(0, 80);
    if (seen.has(key)) {
      // Merge platforms
      const existing = seen.get(key);
      const existingPlatforms = existing.platform.toLowerCase().split(", ");
      const newPlatforms = (src.platform || "").toLowerCase().split(", ");
      const allPlatforms = [...new Set([...existingPlatforms, ...newPlatforms])];
      existing.platform = allPlatforms.join(", ");
      // Keep longer snippet
      if ((src.snippet || "").length > (existing.snippet || "").length) {
        existing.snippet = src.snippet;
      }
    } else {
      seen.set(key, { ...src });
    }
  }
  return [...seen.values()];
}

export default function InsightCard({ insight, index }) {
  const [expanded, setExpanded] = useState(false);
  const sources = dedupeSources(insight.source_details);

  return (
    <div className={`insight-card ${insight.unverified ? "unverified" : ""}`}>
      <div className="insight-header">
        <span className="insight-num">{index + 1}</span>
        <p className="insight-finding">{insight.finding}</p>
        {insight.unverified && <span className="badge-unverified">unverified</span>}
      </div>
      {sources.length > 0 && (
        <div className="insight-sources">
          {sources.map((src, i) => (
            <div key={i} className="source-chip" onClick={() => setExpanded(!expanded)}>
              <span className="source-platform">{
                (src.platform || "").split(", ").map(p => p.charAt(0).toUpperCase() + p.slice(1)).join(" + ")
              }</span>
              <span className="source-year">{src.year}</span>
              <span className="source-title">{src.title}</span>
            </div>
          ))}
          {expanded && sources.map((src, i) => (
            <div key={`detail-${i}`} className="source-detail">
              <p className="source-full-title">{src.title}</p>
              <p className="source-authors">{Array.isArray(src.authors) ? src.authors.join(", ") : src.authors}</p>
              {src.snippet && <p className="source-snippet">"{src.snippet}"</p>}
              {src.url && (
                <a href={src.url} target="_blank" rel="noreferrer" className="source-link">
                  View source
                </a>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

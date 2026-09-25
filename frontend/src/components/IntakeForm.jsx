// De-identified research-context form.

import { useState } from "react";

export default function IntakeForm({ onSubmit }) {
  const [form, setForm] = useState({
    disease: "",
    intent: "",
    location: "",
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!form.disease.trim()) return;
    onSubmit(form);
  };

  return (
    <div className="intake-form-container">
      <div className="intake-card">
        <h2>Curalink</h2>
        <p className="intake-subtitle">AI Medical Research Assistant</p>
        <p className="intake-desc">
          Add de-identified research context below. Curalink uses it throughout
          this session to provide personalized, research-backed answers.
        </p>
        <div className="intake-disclaimer">
          <strong>Do not enter identifiable patient information.</strong> Use a
          condition, research intent, and general location only—never names,
          birth dates, addresses, medical-record numbers, or contact details.
        </div>
        <div className="intake-disclaimer">
          <strong>Not medical advice.</strong> Curalink surfaces published research
          and clinical-trial listings for informational purposes only. It is not a
          diagnosis, treatment recommendation, or a substitute for a qualified
          clinician. Do not use Curalink for emergencies.
        </div>
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label>Disease of Interest *</label>
            <input
              type="text"
              placeholder="e.g. Parkinson's disease"
              value={form.disease}
              onChange={(e) => setForm({ ...form, disease: e.target.value })}
              required
            />
          </div>
          <div className="form-group">
            <label>Additional Query / Intent</label>
            <input
              type="text"
              placeholder="e.g. Deep Brain Stimulation"
              value={form.intent}
              onChange={(e) => setForm({ ...form, intent: e.target.value })}
            />
          </div>
          <div className="form-group">
            <label>Location (for nearby clinical trials)</label>
            <input
              type="text"
              placeholder="e.g. Toronto, Canada"
              value={form.location}
              onChange={(e) => setForm({ ...form, location: e.target.value })}
            />
          </div>
          <button type="submit" className="submit-btn">
            Start Research Session
          </button>
        </form>
      </div>
    </div>
  );
}

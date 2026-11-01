// De-identified research-context form.

import { useState } from "react";

export default function IntakeForm({ onSubmit }) {
  const [form, setForm] = useState({
    patientName: "",
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
        <h2>Start a research session</h2>
        <p className="intake-subtitle">Add context for research-backed answers.</p>
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="patient-name">Patient name (optional)</label>
            <input id="patient-name" name="patientName" type="text" maxLength={100}
              placeholder="Nickname or initials" value={form.patientName}
              onChange={(e) => setForm({ ...form, patientName: e.target.value })} />
          </div>
          <div className="form-group">
            <label htmlFor="research-disease">Disease or condition *</label>
            <input
              id="research-disease"
              type="text"
              placeholder="e.g. Parkinson's disease"
              value={form.disease}
              onChange={(e) => setForm({ ...form, disease: e.target.value })}
              required
            />
          </div>
          <div className="form-group">
            <label htmlFor="research-intent">Research intent</label>
            <input
              id="research-intent"
              type="text"
              placeholder="e.g. Deep Brain Stimulation"
              value={form.intent}
              onChange={(e) => setForm({ ...form, intent: e.target.value })}
            />
          </div>
          <div className="form-group">
            <label htmlFor="research-location">General location</label>
            <input
              id="research-location"
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

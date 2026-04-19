import { useState } from "react";

export default function IntakeForm({ onSubmit }) {
  const [form, setForm] = useState({
    disease: "",
    intent: "",
    location: "",
    patientName: "",
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
          Fill in the consultation details below. This context will be used
          throughout your session to provide personalized, research-backed answers.
        </p>
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
          <div className="form-group">
            <label>Patient Name</label>
            <input
              type="text"
              placeholder="e.g. John Smith"
              value={form.patientName}
              onChange={(e) => setForm({ ...form, patientName: e.target.value })}
            />
          </div>
          <button type="submit" className="submit-btn">
            Start Consultation
          </button>
        </form>
      </div>
    </div>
  );
}

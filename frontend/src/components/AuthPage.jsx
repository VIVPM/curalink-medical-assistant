import { useState } from "react";

export default function AuthPage({ onLogin, onSignup, error }) {
  const [isLogin, setIsLogin] = useState(true);
  const [form, setForm] = useState({ name: "", email: "", password: "" });

  const handleSubmit = (e) => {
    e.preventDefault();
    if (isLogin) {
      onLogin(form.email, form.password);
    } else {
      onSignup(form.name, form.email, form.password);
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-body">
        <h1 className="auth-brand">Curalink</h1>
        <p className="auth-tagline">AI-Powered Medical Research Assistant</p>

        <div className="auth-card">
          <div className="auth-tabs">
            <button
              className={`auth-tab ${isLogin ? "active" : ""}`}
              onClick={() => setIsLogin(true)}
            >
              Login
            </button>
            <button
              className={`auth-tab ${!isLogin ? "active" : ""}`}
              onClick={() => setIsLogin(false)}
            >
              Sign Up
            </button>
          </div>

          <form onSubmit={handleSubmit}>
            {!isLogin && (
              <div className="form-group">
                <label>Name</label>
                <input
                  type="text"
                  placeholder="Your name"
                  value={form.name}
                  onChange={(e) => setForm({ ...form, name: e.target.value })}
                  required={!isLogin}
                />
              </div>
            )}
            <div className="form-group">
              <label>Email</label>
              <input
                type="email"
                placeholder="you@example.com"
                value={form.email}
                onChange={(e) => setForm({ ...form, email: e.target.value })}
                required
              />
            </div>
            <div className="form-group">
              <label>Password</label>
              <input
                type="password"
                placeholder="Min 6 characters"
                value={form.password}
                onChange={(e) => setForm({ ...form, password: e.target.value })}
                required
                minLength={6}
              />
            </div>

            {error && <p className="auth-error">{error}</p>}

            <button type="submit" className="submit-btn">
              {isLogin ? "Login" : "Create Account"}
            </button>
          </form>
        </div>

        <p className="auth-desc">
          Search across PubMed, OpenAlex, and ClinicalTrials.gov in seconds.
          Get structured, source-backed answers grounded in real research — not generic AI responses.
        </p>
      </div>

      <footer className="auth-footer">
        <span>&copy; {new Date().getFullYear()} Curalink. All rights reserved.</span>
      </footer>
    </div>
  );
}

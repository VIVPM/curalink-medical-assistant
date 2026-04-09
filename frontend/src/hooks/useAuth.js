import { useState, useCallback, useEffect } from "react";

const API = `${import.meta.env.VITE_API_URL || ""}/api/auth`;

export default function useAuth() {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(() => localStorage.getItem("token"));
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Check token on mount
  useEffect(() => {
    if (!token) {
      setLoading(false);
      return;
    }
    fetch(`${API}/me`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((res) => res.json())
      .then((data) => {
        if (data.ok) {
          setUser(data.user);
        } else {
          localStorage.removeItem("token");
          setToken(null);
        }
      })
      .catch(() => {
        localStorage.removeItem("token");
        setToken(null);
      })
      .finally(() => setLoading(false));
  }, [token]);

  const signup = useCallback(async (name, email, password) => {
    setError(null);
    const res = await fetch(`${API}/signup`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, email, password }),
    });
    const data = await res.json();
    if (data.ok) {
      localStorage.setItem("token", data.token);
      localStorage.removeItem("activeSessionId");
      setToken(data.token);
      setUser(data.user);
      return true;
    }
    setError(data.error);
    return false;
  }, []);

  const login = useCallback(async (email, password) => {
    setError(null);
    const res = await fetch(`${API}/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
    const data = await res.json();
    if (data.ok) {
      localStorage.setItem("token", data.token);
      localStorage.removeItem("activeSessionId");
      setToken(data.token);
      setUser(data.user);
      return true;
    }
    setError(data.error);
    return false;
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem("token");
    localStorage.removeItem("activeSessionId");
    setToken(null);
    setUser(null);
  }, []);

  // Like logout, but for an expired session: routes back to the login screen
  // with a reason shown, instead of a dead in-app request (UX-5).
  const expire = useCallback(() => {
    localStorage.removeItem("token");
    localStorage.removeItem("activeSessionId");
    setToken(null);
    setUser(null);
    setError("Your session expired. Please log in again.");
  }, []);

  return { user, token, loading, error, signup, login, logout, expire };
}

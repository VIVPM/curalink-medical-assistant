// Frontend authentication and account lifecycle state.

import { useState, useCallback, useEffect } from "react";

const API_ROOT = `${import.meta.env.VITE_API_URL || ""}/api`;
const API = `${API_ROOT}/auth`;

export default function useAuth() {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(() => localStorage.getItem("token"));
  const [loading, setLoading] = useState(() => Boolean(localStorage.getItem("token")));
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!token) return;
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

  const signup = useCallback(async (name, email, password, acceptTerms) => {
    setError(null);
    const res = await fetch(`${API}/signup`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, email, password, acceptTerms }),
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

  const deleteAccount = useCallback(async () => {
    setError(null);
    const res = await fetch(`${API_ROOT}/account`, {
      method: "DELETE",
      headers: { Authorization: `Bearer ${token}` },
    });
    if (!res.ok) {
      setError("Account deletion failed. Please try again.");
      return false;
    }
    localStorage.removeItem("token");
    localStorage.removeItem("activeSessionId");
    setToken(null);
    setUser(null);
    return true;
  }, [token]);

  const expire = useCallback(() => {
    localStorage.removeItem("token");
    localStorage.removeItem("activeSessionId");
    setToken(null);
    setUser(null);
    setError("Your session expired. Please log in again.");
  }, []);

  return { user, token, loading, error, signup, login, logout, deleteAccount, expire };
}

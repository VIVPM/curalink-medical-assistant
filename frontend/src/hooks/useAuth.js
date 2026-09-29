// Frontend authentication scoped to a browser tab with renewable access tokens.

import { useState, useCallback, useEffect } from "react";
import { authFetch, clearPrivateSession, readSession, refreshAccess, saveSession } from "../session";

const API_ROOT = `${import.meta.env.VITE_API_URL || ""}/api`;
const API = `${API_ROOT}/auth`;

export default function useAuth() {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(() => readSession("token"));
  const [loading, setLoading] = useState(() => Boolean(readSession("token") || readSession("refreshToken")));
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!token && !readSession("refreshToken")) return;
    let cancelled = false;
    let retryTimer;

    const checkSession = (attempt) => {
      authFetch(`${API}/me`)
        .then(async (res) => {
          if (cancelled) return;
          if (res.status === 401) {
            clearPrivateSession();
            setToken(null);
            setLoading(false);
            return;
          }
          const data = await res.json();
          if (!res.ok || !data.ok) throw new Error(`session check failed: ${res.status}`);
          setUser(data.user);
          setToken(readSession("token"));
          setLoading(false);
        })
        .catch(() => {
          if (cancelled) return;
          if (attempt < 4) {
            retryTimer = setTimeout(() => checkSession(attempt + 1), 1000 * 2 ** attempt);
            return;
          }
          setError("Could not verify your session. Please refresh the page.");
          setLoading(false);
        });
    };

    checkSession(0);
    return () => {
      cancelled = true;
      clearTimeout(retryTimer);
    };
  }, [token]);

  useEffect(() => {
    if (!user) return;
    const interval = setInterval(refreshAccess, 45 * 60 * 1000);
    const onVisible = () => { if (document.visibilityState === "visible") refreshAccess(); };
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      clearInterval(interval);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, [user]);

  const signup = useCallback(async (name, email, password, acceptTerms) => {
    setError(null);
    const res = await fetch(`${API}/signup`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, email, password, acceptTerms }),
    });
    const data = await res.json();
    if (data.ok) {
      clearPrivateSession();
      saveSession("token", data.token);
      saveSession("refreshToken", data.refreshToken);
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
      clearPrivateSession();
      saveSession("token", data.token);
      saveSession("refreshToken", data.refreshToken);
      setToken(data.token);
      setUser(data.user);
      return true;
    }
    setError(data.error);
    return false;
  }, []);

  const logout = useCallback(async () => {
    try { await authFetch(`${API}/logout`, { method: "POST" }); } catch (error) { void error; }
    clearPrivateSession();
    setToken(null);
    setUser(null);
  }, []);

  const deleteAccount = useCallback(async () => {
    setError(null);
    const res = await authFetch(`${API_ROOT}/account`, { method: "DELETE" });
    if (!res.ok) {
      setError("Account deletion failed. Please try again.");
      return false;
    }
    clearPrivateSession();
    setToken(null);
    setUser(null);
    return true;
  }, []);

  const expire = useCallback(() => {
    clearPrivateSession();
    setToken(null);
    setUser(null);
    setError("Your session expired. Please log in again.");
  }, []);

  return { user, token, loading, error, signup, login, logout, deleteAccount, expire };
}

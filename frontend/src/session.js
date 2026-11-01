// Keeps login scoped to the current browser tab; provider credentials stay on the server.

const API = `${import.meta.env.VITE_API_URL}/api`;
let refreshPromise = null;

try {
  localStorage.removeItem("token");
  localStorage.removeItem("activeSessionId");
  sessionStorage.removeItem("ownKeys");
} catch (error) {
  void error;
}

export function readSession(name) {
  try { return sessionStorage.getItem(name); } catch { return null; }
}

export function saveSession(name, value) {
  try { sessionStorage.setItem(name, value); } catch { return false; }
  return true;
}

export function removeSession(name) {
  try { sessionStorage.removeItem(name); } catch (error) { void error; }
}

export function clearPrivateSession() {
  for (const key of ["token", "refreshToken", "activeSessionId", "ownKeys"]) removeSession(key);
}

export async function refreshAccess() {
  if (refreshPromise) return refreshPromise;
  const refreshToken = readSession("refreshToken");
  if (!refreshToken) return null;
  refreshPromise = fetch(`${API}/auth/refresh`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ refreshToken }),
  }).then(async (response) => {
    if (!response.ok) {
      if (response.status === 401) clearPrivateSession();
      return null;
    }
    const data = await response.json();
    saveSession("token", data.token);
    saveSession("refreshToken", data.refreshToken);
    return data.token;
  }).catch(() => null).finally(() => { refreshPromise = null; });
  return refreshPromise;
}

export async function authFetch(url, options = {}) {
  const makeRequest = (token) => fetch(url, {
    ...options,
    headers: { ...options.headers, ...(token ? { Authorization: `Bearer ${token}` } : {}) },
  });
  const response = await makeRequest(readSession("token"));
  if (response.status !== 401 || !readSession("refreshToken")) return response;
  const token = await refreshAccess();
  return token ? makeRequest(token) : response;
}

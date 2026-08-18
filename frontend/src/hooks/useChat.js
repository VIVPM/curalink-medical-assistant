import { useState, useCallback, useRef } from "react";

const API = `${import.meta.env.VITE_API_URL || ""}/api`;

function getAuthHeaders() {
  const token = localStorage.getItem("token");
  const headers = { "Content-Type": "application/json" };
  if (token) headers["Authorization"] = `Bearer ${token}`;
  return headers;
}

export default function useChat({ onAuthExpired } = {}) {
  const [sessions, setSessions] = useState([]);
  const [activeSession, setActiveSession] = useState(null);
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [streamStatus, setStreamStatus] = useState(null);
  const [pipelineStage, setPipelineStage] = useState(null); // current stage name from SSE
  const [retrievalCounts, setRetrievalCounts] = useState(null); // live retrieval counts
  const [waking, setWaking] = useState(false); // true during cold-start wait for a sleeping free-tier server
  const abortRef = useRef(null);

  const fetchSessions = useCallback(async () => {
    const res = await fetch(`${API}/sessions`, { headers: getAuthHeaders() });
    const data = await res.json();
    if (data.ok) setSessions(data.sessions);
  }, []);

  const createSession = useCallback(async (form) => {
    const res = await fetch(`${API}/session`, {
      method: "POST",
      headers: getAuthHeaders(),
      body: JSON.stringify(form),
    });
    const data = await res.json();
    if (data.ok) {
      setActiveSession(data.session);
      localStorage.setItem("activeSessionId", data.session._id);
      setMessages([]);
      fetchSessions();
      return data.session;
    }
    return null;
  }, [fetchSessions]);

  const loadSession = useCallback(async (id) => {
    const res = await fetch(`${API}/session/${id}`, { headers: getAuthHeaders() });
    const data = await res.json();
    if (data.ok) {
      setActiveSession(data.session);
      localStorage.setItem("activeSessionId", data.session._id);
      setMessages(data.messages);
    }
  }, []);

  const sendMessage = useCallback(async (text) => {
    if (!activeSession || loading) return;

    const userMsg = { role: "user", content: text, _id: Date.now().toString() };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);
    setStreamStatus("Starting pipeline...");
    setPipelineStage("starting");
    setRetrievalCounts(null);

    const assistantId = (Date.now() + 1).toString();
    let gotResult = false;

    // Free-tier services sleep after ~15 min idle; the first request then waits
    // 30-60s for a cold boot. Show an honest notice if headers don't arrive fast.
    let wakeTimer = setTimeout(() => setWaking(true), 5000);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch(`${API}/chat/stream`, {
        method: "POST",
        headers: getAuthHeaders(),
        signal: controller.signal,
        body: JSON.stringify({
          sessionId: activeSession._id,
          message: text,
        }),
      });

      // Response headers arrived -> server is awake; drop the cold-start notice.
      clearTimeout(wakeTimer);
      setWaking(false);

      // An error response (401/404/5xx) is JSON/HTML, not SSE. Feeding it to the
      // parser fails silently and looks like "no response" (BUG-2) — handle it.
      if (!res.ok) {
        gotResult = true;
        if (res.status === 401) {
          // Session expired — route to the login screen with a reason (UX-5)
          // rather than a dead request. Fall back to an in-chat notice.
          if (onAuthExpired) {
            onAuthExpired();
          } else {
            localStorage.removeItem("token");
            setMessages((prev) => [
              ...prev,
              { role: "assistant", content: "Your session expired. Please refresh the page and log in again.", _id: assistantId, error: true },
            ]);
          }
        } else {
          const msg =
            res.status === 402
              ? "You're out of questions (credits) for this demo."
              : res.status === 404
              ? "This session no longer exists."
              : `The server returned an error (${res.status}). Please try again.`;
          setMessages((prev) => [
            ...prev,
            { role: "assistant", content: msg, _id: assistantId, error: true },
          ]);
        }
      }

      const reader = res.ok ? res.body.getReader() : null;
      const decoder = new TextDecoder();
      let buffer = "";
      let currentEvent = null;

      while (reader) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("event: ")) {
            currentEvent = line.slice(7);
          } else if (line.startsWith("data: ") && currentEvent) {
            const data = line.slice(6);
            if (currentEvent === "status") {
              try {
                const info = JSON.parse(data);
                setStreamStatus(info.message || info.stage);
                if (info.stage) setPipelineStage(info.stage);
                if (info.retrieval_counts) setRetrievalCounts(info.retrieval_counts);
              } catch { /* ignore malformed SSE data */ }
            } else if (currentEvent === "metadata") {
              try {
                const meta = JSON.parse(data);
                gotResult = true;
                setMessages((prev) => [
                  ...prev,
                  {
                    role: "assistant",
                    content: meta.overview || "",
                    structuredResponse: meta,
                    _id: assistantId,
                  },
                ]);
              } catch { /* ignore malformed SSE data */ }
            } else if (currentEvent === "error") {
              try {
                const errData = JSON.parse(data);
                gotResult = true;
                setMessages((prev) => [
                  ...prev,
                  {
                    role: "assistant",
                    content: errData.error || "Pipeline error",
                    _id: assistantId,
                    error: true,
                  },
                ]);
              } catch { /* ignore malformed SSE data */ }
            }
            currentEvent = null;
          }
        }
      }

      if (!gotResult) {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: "The assistant didn't respond. Please try again.",
            _id: assistantId,
            error: true,
          },
        ]);
      }
    } catch (err) {
      clearTimeout(wakeTimer);
      if (err.name === "AbortError") {
        // User pressed Stop — not an error.
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: "⏹ Generation stopped.", _id: assistantId },
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: "Could not reach the server. Please check that all services are running and try again.",
            _id: assistantId,
            error: true,
          },
        ]);
      }
    }

    clearTimeout(wakeTimer);
    setWaking(false);
    setLoading(false);
    setStreamStatus(null);
    setPipelineStage(null);
    setRetrievalCounts(null);
    abortRef.current = null;
    fetchSessions();
  }, [activeSession, loading, fetchSessions, onAuthExpired]);

  const stopGeneration = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  return {
    sessions,
    activeSession,
    messages,
    loading,
    streamStatus,
    pipelineStage,
    retrievalCounts,
    waking,
    fetchSessions,
    createSession,
    loadSession,
    sendMessage,
    stopGeneration,
    setActiveSession,
    setMessages,
  };
}

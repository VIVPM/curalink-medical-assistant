// Frontend session, streaming chat, and credit state.

import { useState, useCallback, useRef } from "react";
import { authFetch, getOwnKeys, ownKeyHeaders, removeSession, saveSession } from "../session";

const API = `${import.meta.env.VITE_API_URL || ""}/api`;

function getAuthHeaders() {
  return { "Content-Type": "application/json" };
}

export default function useChat({ onAuthExpired, onOwnKeysRejected } = {}) {
  const [sessions, setSessions] = useState([]);
  const [activeSession, setActiveSession] = useState(null);
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [streamStatus, setStreamStatus] = useState(null);
  const [pipelineStage, setPipelineStage] = useState(null);
  const [retrievalCounts, setRetrievalCounts] = useState(null);
  const [waking, setWaking] = useState(false);
  const [credits, setCredits] = useState(null);
  const abortRef = useRef(null);

  const fetchCredits = useCallback(async () => {
    try {
      const res = await authFetch(`${API}/account/credits`, { headers: getAuthHeaders() });
      const data = await res.json();
      if (data.ok) setCredits(data);
    } catch (error) {
      void error;
    }
  }, []);

  const fetchSessions = useCallback(async () => {
    const res = await authFetch(`${API}/sessions`, { headers: getAuthHeaders() });
    const data = await res.json();
    if (data.ok) setSessions(data.sessions);
  }, []);

  const createSession = useCallback(async (form) => {
    const res = await authFetch(`${API}/session`, {
      method: "POST",
      headers: getAuthHeaders(),
      body: JSON.stringify(form),
    });
    const data = await res.json();
    if (data.ok) {
      setActiveSession(data.session);
      saveSession("activeSessionId", data.session._id);
      setMessages([]);
      fetchSessions();
      return data.session;
    }
    return null;
  }, [fetchSessions]);

  const loadSession = useCallback(async (id) => {
    const res = await authFetch(`${API}/session/${id}`, { headers: getAuthHeaders() });
    const data = await res.json();
    if (data.ok) {
      setActiveSession(data.session);
      saveSession("activeSessionId", data.session._id);
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

    let wakeTimer = setTimeout(() => setWaking(true), 5000);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await authFetch(`${API}/chat/stream`, {
        method: "POST",
        headers: { ...getAuthHeaders(), ...ownKeyHeaders() },
        signal: controller.signal,
        body: JSON.stringify({
          sessionId: activeSession._id,
          message: text,
        }),
      });

      clearTimeout(wakeTimer);
      setWaking(false);

      if (!res.ok) {
        gotResult = true;
        if (res.status === 401) {
          setActiveSession(null);
          setMessages([]);
          removeSession("activeSessionId");

          if (onAuthExpired) {
            onAuthExpired();
          } else {
            removeSession("token");
            setMessages((prev) => [
              ...prev,
              { role: "assistant", content: "Your session expired. Please refresh the page and log in again.", _id: assistantId, error: true },
            ]);
          }
        } else {
          const rejectedOwnKeys = res.status === 400 && Boolean(getOwnKeys());
          if (rejectedOwnKeys) {
            removeSession("ownKeys");
            onOwnKeysRejected?.();
          }
          const msg =
            rejectedOwnKeys
              ? "API keys were rejected. Update them in Settings."
              : res.status === 402
              ? "Daily limit reached. Resets at midnight UTC."
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
              } catch (error) {
                void error;
              }
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
              } catch (error) {
                void error;
              }
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
              } catch (error) {
                void error;
              }
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
    fetchCredits();
  }, [activeSession, loading, fetchSessions, fetchCredits, onAuthExpired, onOwnKeysRejected]);

  const deleteSession = useCallback(async (id) => {
    const res = await authFetch(`${API}/session/${id}`, {
      method: "DELETE",
      headers: getAuthHeaders(),
    });
    if (res.status === 401) {
      onAuthExpired?.();
      return false;
    }
    if (!res.ok) return false;

    setSessions((current) => current.filter((session) => session._id !== id));
    if (activeSession?._id === id) {
      setActiveSession(null);
      setMessages([]);
      removeSession("activeSessionId");
    }
    return true;
  }, [activeSession, onAuthExpired]);

  const resetChat = useCallback(() => {
    abortRef.current?.abort();
    setSessions([]);
    setActiveSession(null);
    setMessages([]);
    setLoading(false);
    setStreamStatus(null);
    setPipelineStage(null);
    setRetrievalCounts(null);
    setWaking(false);
    removeSession("activeSessionId");
  }, []);

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
    credits,
    fetchSessions,
    fetchCredits,
    createSession,
    loadSession,
    deleteSession,
    sendMessage,
    stopGeneration,
    resetChat,
    setActiveSession,
    setMessages,
  };
}

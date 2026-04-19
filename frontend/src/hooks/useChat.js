import { useState, useCallback, useRef } from "react";

const API = `${import.meta.env.VITE_API_URL || ""}/api`;

function getAuthHeaders() {
  const token = localStorage.getItem("token");
  const headers = { "Content-Type": "application/json" };
  if (token) headers["Authorization"] = `Bearer ${token}`;
  return headers;
}

export default function useChat() {
  const [sessions, setSessions] = useState([]);
  const [activeSession, setActiveSession] = useState(null);
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [streamStatus, setStreamStatus] = useState(null);
  const [pipelineStage, setPipelineStage] = useState(null); // current stage name from SSE
  const [retrievalCounts, setRetrievalCounts] = useState(null); // live retrieval counts
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
      setMessages(data.messages);
    }
  }, []);

  const sendMessage = useCallback(async (text) => {
    if (!activeSession || loading) return;

    const userMsg = { role: "user", content: text, _id: Date.now().toString() };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);
    setStreamStatus("Running pipeline...");
    setPipelineStage("processing");
    setRetrievalCounts(null);

    const assistantId = (Date.now() + 1).toString();

    try {
      const res = await fetch(`${API}/chat`, {
        method: "POST",
        headers: getAuthHeaders(),
        body: JSON.stringify({
          sessionId: activeSession._id,
          message: text,
        }),
      });

      const data = await res.json();

      if (!data.ok) {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: data.error || "Pipeline error",
            _id: assistantId,
            error: true,
          },
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: data.response?.overview || "",
            structuredResponse: data.response,
            _id: data.assistantMessage?._id || assistantId,
          },
        ]);
      }
    } catch (err) {
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

    setLoading(false);
    setStreamStatus(null);
    setPipelineStage(null);
    setRetrievalCounts(null);
    fetchSessions();
  }, [activeSession, loading, fetchSessions]);

  return {
    sessions,
    activeSession,
    messages,
    loading,
    streamStatus,
    pipelineStage,
    retrievalCounts,
    fetchSessions,
    createSession,
    loadSession,
    sendMessage,
    setActiveSession,
    setMessages,
  };
}

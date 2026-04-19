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
    setStreamStatus("Starting pipeline...");
    setPipelineStage("starting");
    setRetrievalCounts(null);

    const assistantId = (Date.now() + 1).toString();
    let gotResult = false;

    try {
      const res = await fetch(`${API}/chat/stream`, {
        method: "POST",
        headers: getAuthHeaders(),
        body: JSON.stringify({
          sessionId: activeSession._id,
          message: text,
        }),
      });

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let currentEvent = null;

      while (true) {
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
              } catch {}
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
              } catch {}
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
              } catch {}
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

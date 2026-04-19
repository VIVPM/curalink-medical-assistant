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

    // Add user message immediately
    const userMsg = { role: "user", content: text, _id: Date.now().toString() };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);
    setStreamStatus("Starting pipeline...");
    setPipelineStage("starting");
    setRetrievalCounts(null);

    // Placeholder for assistant response
    const assistantId = (Date.now() + 1).toString();
    const assistantMsg = {
      role: "assistant",
      content: "",
      structuredResponse: null,
      _id: assistantId,
      streaming: true,
    };
    setMessages((prev) => [...prev, assistantMsg]);

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
      let tokens = "";

      let gotMetadata = false;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        let currentEvent = null;
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
            } else if (currentEvent === "token") {
              // Collect tokens silently (LLM outputs JSON, not readable text).
              // Progress bar handles the loading UX — no need to render raw JSON.
              try {
                tokens += JSON.parse(data);
              } catch {
                tokens += data;
              }
            } else if (currentEvent === "metadata") {
              try {
                const meta = JSON.parse(data);
                gotMetadata = true;
                setMessages((prev) =>
                  prev.map((m) =>
                    m._id === assistantId
                      ? { ...m, structuredResponse: meta, streaming: false, content: meta.overview || tokens }
                      : m
                  )
                );
              } catch {}
            } else if (currentEvent === "error") {
              try {
                const errData = JSON.parse(data);
                setMessages((prev) =>
                  prev.map((m) =>
                    m._id === assistantId
                      ? { ...m, content: errData.error || "Pipeline error", streaming: false, error: true }
                      : m
                  )
                );
              } catch {}
            } else if (currentEvent === "done") {
              setStreamStatus(null);
              setPipelineStage(null);
            }
            currentEvent = null;
          }
        }
      }

      // Stream ended. If no metadata/error event ever arrived, the pipeline
      // died silently (cold start, OOM, upstream hangup). Mark the message
      // as errored so the blinking cursor doesn't hang forever.
      if (!gotMetadata) {
        setMessages((prev) =>
          prev.map((m) =>
            m._id === assistantId && m.streaming
              ? { ...m, content: "The assistant didn't respond. Please try again.", streaming: false, error: true }
              : m
          )
        );
        setStreamStatus(null);
        setPipelineStage(null);
      }
    } catch (err) {
      setMessages((prev) =>
        prev.map((m) =>
          m._id === assistantId
            ? { ...m, content: "Could not reach the server. Please check that all services are running and try again.", streaming: false, error: true }
            : m
        )
      );
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

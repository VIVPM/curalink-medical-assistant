import { useState, useRef, useEffect, useMemo } from "react";
import StructuredResponse from "./StructuredResponse";
import PipelinePanel from "./PipelinePanel";
import PipelineProgress from "./PipelineProgress";

/* ---------- Suggested questions generator ---------- */
function getSuggestedQuestions(disease, intent) {
  if (!disease) return [];

  const questions = [
    `What are the latest clinical trials for ${disease}?`,
    `What treatment options are being researched for ${disease}?`,
    `What are the common symptoms and risk factors of ${disease}?`,
    `Are there any recent breakthroughs in ${disease} research?`,
    `What is the prognosis for patients with ${disease}?`,
    `What lifestyle changes are recommended for ${disease}?`,
  ];

  // Add intent-specific questions
  if (intent) {
    questions.unshift(`What does current research say about ${intent} for ${disease}?`);
  }

  // Return first 4 suggestions
  return questions.slice(0, 4);
}

export default function ChatView({
  session,
  messages,
  loading,
  streamStatus,
  pipelineStage,
  retrievalCounts,
  onSend,
  onBack,
}) {
  const [input, setInput] = useState("");
  const [panelCollapsed, setPanelCollapsed] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamStatus, pipelineStage]);

  // Extract the latest pipelineMeta from the most recent assistant message
  const latestMeta = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      const meta = messages[i]?.structuredResponse?.pipelineMeta;
      if (meta) return meta;
    }
    return null;
  }, [messages]);

  // Generate contextual suggestions
  const suggestions = useMemo(() => {
    return getSuggestedQuestions(
      session.staticContext?.disease,
      session.staticContext?.intent
    );
  }, [session.staticContext?.disease, session.staticContext?.intent]);

  // Hide static suggestions once any response has follow-up questions
  const hasFollowUps = messages.some(
    (m) => m.structuredResponse?.follow_up_questions?.length >= 2
  );
  const showSuggestions = !loading && suggestions.length > 0 && !hasFollowUps;

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;
    onSend(input.trim());
    setInput("");
  };

  const handleSuggestionClick = (question) => {
    if (loading) return;
    onSend(question);
  };

  return (
    <div className="chat-view-wrapper">
      <div className="chat-view">
        <div className="chat-header">
          <button className="back-btn" onClick={onBack}>
            &larr;
          </button>
          <div className="chat-header-info">
            <h3>{session.title}</h3>
            <span className="chat-context">
              {session.staticContext?.disease}
              {session.staticContext?.intent && ` | ${session.staticContext.intent}`}
              {session.staticContext?.location && ` | ${session.staticContext.location}`}
            </span>
          </div>
        </div>

        <div className="chat-messages">
          {messages.map((msg) => {
            // Hide the empty streaming placeholder while progress bar is visible
            if (msg.streaming && !msg.content && pipelineStage) return null;

            return (
              <div key={msg._id} className={`message ${msg.role}`}>
                <div className="message-role">
                  <span className="role-icon">{msg.role === "user" ? "\uD83D\uDC64" : "\uD83E\uDD16"}</span>
                  {msg.role === "user" ? "You" : "Assistant"}
                </div>
                {msg.error ? (
                  <div className="message-error">
                    <span className="error-icon">!</span>
                    <p>{msg.content}</p>
                  </div>
                ) : msg.role === "assistant" && msg.structuredResponse ? (
                  <StructuredResponse data={msg.structuredResponse} onFollowUp={!loading ? onSend : undefined} />
                ) : (
                  <div className="message-content">
                    {msg.content}
                    {msg.streaming && <span className="cursor">|</span>}
                  </div>
                )}
              </div>
            );
          })}

          {pipelineStage && <PipelineProgress stage={pipelineStage} retrievalCounts={retrievalCounts} />}

          <div ref={bottomRef} />
        </div>

        <form className="chat-input-form" onSubmit={handleSubmit}>
          <input
            type="text"
            placeholder="Ask a follow-up question..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={loading}
          />
          <button type="submit" disabled={loading || !input.trim()}>
            Send
          </button>
        </form>

        {showSuggestions && (
          <div className="suggested-questions">
            <span className="sq-label">Try asking:</span>
            <div className="sq-chips">
              {suggestions.map((q, i) => (
                <button
                  key={i}
                  className="sq-chip"
                  onClick={() => handleSuggestionClick(q)}
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        <div className="chat-footer-disclaimer">
          Responses are AI-generated from retrieved research papers and clinical trials. Always verify findings with original sources before making medical decisions.
        </div>
      </div>

      {latestMeta && (
        <PipelinePanel
          meta={latestMeta}
          collapsed={panelCollapsed}
          onToggle={() => setPanelCollapsed((v) => !v)}
        />
      )}
    </div>
  );
}

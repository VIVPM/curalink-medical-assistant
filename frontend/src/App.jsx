import { useEffect, useState } from "react";
import useAuth from "./hooks/useAuth";
import useChat from "./hooks/useChat";
import AuthPage from "./components/AuthPage";
import Sidebar from "./components/Sidebar";
import IntakeForm from "./components/IntakeForm";
import ChatView from "./components/ChatView";
import "./App.css";

export default function App() {
  const { user, loading: authLoading, error: authError, signup, login, logout } = useAuth();

  const {
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
  } = useChat();

  const [showForm, setShowForm] = useState(
    () => !localStorage.getItem("activeSessionId")
  );
  // True while we're fetching a stored session on page refresh — prevents
  // the intake form from flashing before the session loads.
  const [rehydrating, setRehydrating] = useState(
    () => !!localStorage.getItem("activeSessionId")
  );

  useEffect(() => {
    if (user) fetchSessions();
  }, [user, fetchSessions]);

  useEffect(() => {
    if (authLoading) return;
    if (!user) {
      setRehydrating(false);
      return;
    }
    const lastId = localStorage.getItem("activeSessionId");
    if (!lastId) {
      setActiveSession(null);
      setMessages([]);
      setShowForm(true);
      setRehydrating(false);
      return;
    }
    loadSession(lastId)
      .then(() => setShowForm(false))
      .catch(() => setShowForm(true))
      .finally(() => setRehydrating(false));
  }, [authLoading, user, loadSession, setActiveSession, setMessages]);

  if (authLoading || (user && rehydrating)) {
    return (
      <div className="loading-screen">
        <div className="spinner" />
        <p>Loading...</p>
      </div>
    );
  }

  if (!user) {
    return <AuthPage onLogin={login} onSignup={signup} error={authError} />;
  }

  const handleNewSession = () => {
    setActiveSession(null);
    setMessages([]);
    setShowForm(true);
    localStorage.removeItem("activeSessionId");
  };

  const handleFormSubmit = async (form) => {
    const session = await createSession(form);
    if (session) setShowForm(false);
  };

  const handleSelectSession = async (id) => {
    await loadSession(id);
    setShowForm(false);
  };

  return (
    <div className="app">
      <Sidebar
        sessions={sessions}
        activeId={activeSession?._id}
        onSelect={handleSelectSession}
        onNew={handleNewSession}
        userName={user.name}
        onLogout={logout}
      />
      <main className="main-content">
        {showForm || !activeSession ? (
          <IntakeForm onSubmit={handleFormSubmit} />
        ) : (
          <ChatView
            session={activeSession}
            messages={messages}
            loading={loading}
            streamStatus={streamStatus}
            pipelineStage={pipelineStage}
            retrievalCounts={retrievalCounts}
            onSend={sendMessage}
            onBack={handleNewSession}
          />
        )}
      </main>
    </div>
  );
}

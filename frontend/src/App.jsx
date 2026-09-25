import { useEffect, useState } from "react";
import useAuth from "./hooks/useAuth";
import useChat from "./hooks/useChat";
import AuthPage from "./components/AuthPage";
import LandingPage from "./components/LandingPage";
import LegalPage from "./components/LegalPage";
import Sidebar from "./components/Sidebar";
import IntakeForm from "./components/IntakeForm";
import ChatView from "./components/ChatView";
import "./App.css";

export default function App() {
  const { user, loading: authLoading, error: authError, signup, login, logout, deleteAccount, expire } = useAuth();

  const {
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
    setActiveSession,
    setMessages,
  } = useChat({ onAuthExpired: expire });

  const [showForm, setShowForm] = useState(
    () => !localStorage.getItem("activeSessionId")
  );
  // True while we're fetching a stored session on page refresh — prevents
  // the intake form from flashing before the session loads.
  const [rehydrating, setRehydrating] = useState(
    () => !!localStorage.getItem("activeSessionId")
  );
  const [showAuth, setShowAuth] = useState(false);
  const [authMode, setAuthMode] = useState("login");
  const [legalPage, setLegalPage] = useState(null);

  useEffect(() => {
    if (user) { fetchSessions(); fetchCredits(); }
  }, [user, fetchSessions, fetchCredits]);

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

  if (legalPage) {
    return <LegalPage type={legalPage} onBack={() => setLegalPage(null)} />;
  }

  if (authLoading || (user && rehydrating)) {
    return (
      <div className="loading-screen">
        <div className="spinner" />
        <p>Loading...</p>
      </div>
    );
  }

  if (!user) {
    // Landing first; the auth screen appears when they choose to sign in / get
    // started, or when a session expiry (authError) needs a re-login.
    if (!showAuth && !authError) {
      return (
        <LandingPage
          onGetStarted={() => { setAuthMode("signup"); setShowAuth(true); }}
          onSignIn={() => { setAuthMode("login"); setShowAuth(true); }}
          onPrivacy={() => setLegalPage("privacy")}
          onTerms={() => setLegalPage("terms")}
        />
      );
    }
    return (
      <AuthPage
        onLogin={login}
        onSignup={signup}
        error={authError}
        initialMode={authMode}
        onBack={() => setShowAuth(false)}
        onPrivacy={() => setLegalPage("privacy")}
        onTerms={() => setLegalPage("terms")}
      />
    );
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

  const handleDeleteSession = async (id) => {
    if (!window.confirm("Delete this research session and all of its messages?")) return;
    const deleted = await deleteSession(id);
    if (!deleted) window.alert("Session deletion failed. Please try again.");
  };

  const handleDeleteAccount = async () => {
    if (!window.confirm("Permanently delete your account and all stored research sessions? This cannot be undone.")) return;
    const deleted = await deleteAccount();
    if (deleted) setShowAuth(false);
    else window.alert("Account deletion failed. Please try again.");
  };

  return (
    <div className="app">
      <Sidebar
        sessions={sessions}
        activeId={activeSession?._id}
        onSelect={handleSelectSession}
        onDelete={handleDeleteSession}
        onNew={handleNewSession}
        userName={user.name}
        credits={credits}
        onLogout={logout}
        onDeleteAccount={handleDeleteAccount}
        onPrivacy={() => setLegalPage("privacy")}
        onTerms={() => setLegalPage("terms")}
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
            waking={waking}
            onSend={sendMessage}
            onStop={stopGeneration}
            onBack={handleNewSession}
          />
        )}
      </main>
    </div>
  );
}

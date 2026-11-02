// Root application flow for landing, authentication, legal, and research views.

import { useCallback, useEffect, useState } from "react";
import useAuth from "./hooks/useAuth";
import useChat from "./hooks/useChat";
import AuthPage from "./components/AuthPage";
import LandingPage from "./components/LandingPage";
import LegalPage from "./components/LegalPage";
import Sidebar from "./components/Sidebar";
import IntakeForm from "./components/IntakeForm";
import ChatView from "./components/ChatView";
import ApiKeySettings from "./components/ApiKeySettings";
import { readSession, removeSession } from "./session";
import "./App.css";

export default function App() {
  const { user, loading: authLoading, error: authError, signup, login, logout, deleteAccount, expire } = useAuth();
  const [showKeySettings, setShowKeySettings] = useState(false);
  const handleOwnKeysRejected = useCallback(() => setShowKeySettings(true), []);

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
    resetChat,
    setActiveSession,
    setMessages,
  } = useChat({ onAuthExpired: expire, onOwnKeysRejected: handleOwnKeysRejected });

  const [showForm, setShowForm] = useState(
    () => !readSession("activeSessionId")
  );

  const [rehydrating, setRehydrating] = useState(
    () => Boolean(readSession("token") && readSession("activeSessionId"))
  );
  const [showAuth, setShowAuth] = useState(false);
  const [authMode, setAuthMode] = useState("login");
  const [legalPage, setLegalPage] = useState(null);

  useEffect(() => {
    if (user) { fetchSessions(); fetchCredits(); }
  }, [user, fetchSessions, fetchCredits]);

  useEffect(() => {
    if (authLoading) return;
    if (!user) return;
    const lastId = readSession("activeSessionId");
    if (!lastId) return;
    loadSession(lastId)
      .then(() => setShowForm(false))
      .catch(() => setShowForm(true))
      .finally(() => setRehydrating(false));
  }, [authLoading, user, loadSession]);

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
    removeSession("activeSessionId");
  };

  const handleLogout = () => {
    resetChat();
    setShowForm(true);
    logout();
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
    if (deleted) {
      resetChat();
      setShowForm(true);
      setShowAuth(false);
    } else window.alert("Account deletion failed. Please try again.");
  };

  return (
    <div className="app">
      <Sidebar
        sessions={sessions}
        activeId={activeSession?._id}
        onSelect={handleSelectSession}
        onDelete={handleDeleteSession}
        onNew={handleNewSession}
        userName={user.email}
        onSettings={() => setShowKeySettings(true)}
        onLogout={handleLogout}
      />
      <main className="main-content">
        <header className="app-navbar">
          <h1>Curalink</h1>
          {credits && !credits.unlimited && <span className={`navbar-credits${credits.remaining === 0 ? " navbar-credits-empty" : ""}`}
            role="status" title="Daily credits reset at midnight UTC">
            <strong>{credits.remaining}</strong> / {credits.cap} credits<span className="navbar-credit-detail"> left today</span>
          </span>}
        </header>
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
        <footer className="research-footer">
          <strong>Research only, not medical advice.</strong> Published research and trial listings are not a diagnosis,
          treatment recommendation or substitute for a clinician. Not for emergencies.
        </footer>
      </main>
      {showKeySettings && (
        <ApiKeySettings
          credits={credits}
          onPrivacy={() => { setShowKeySettings(false); setLegalPage("privacy"); }}
          onTerms={() => { setShowKeySettings(false); setLegalPage("terms"); }}
          onDeleteAccount={() => { setShowKeySettings(false); handleDeleteAccount(); }}
          onChange={fetchCredits}
          onClose={() => setShowKeySettings(false)}
        />
      )}
    </div>
  );
}

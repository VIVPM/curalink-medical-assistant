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
import { getOwnKeys, readSession, removeSession, saveSession } from "./session";
import "./App.css";

export default function App() {
  const { user, loading: authLoading, error: authError, signup, login, logout, deleteAccount, expire } = useAuth();
  const [ownKeys, setOwnKeys] = useState(getOwnKeys);
  const [showKeySettings, setShowKeySettings] = useState(false);
  const clearOwnKeyState = useCallback(() => setOwnKeys(null), []);
  const handleExpire = useCallback(() => {
    setOwnKeys(null);
    expire();
  }, [expire]);

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
  } = useChat({ onAuthExpired: handleExpire, onOwnKeysRejected: clearOwnKeyState });

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

  const saveOwnKeys = (keys) => {
    if (!saveSession("ownKeys", JSON.stringify(keys))) return false;
    setOwnKeys(keys);
    fetchCredits();
    return true;
  };

  const removeOwnKeys = () => {
    removeSession("ownKeys");
    setOwnKeys(null);
    fetchCredits();
  };

  const handleNewSession = () => {
    setActiveSession(null);
    setMessages([]);
    setShowForm(true);
    removeSession("activeSessionId");
  };

  const handleLogout = () => {
    resetChat();
    setOwnKeys(null);
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
      setOwnKeys(null);
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
        userName={user.name}
        credits={credits}
        usingOwnKeys={Boolean(ownKeys)}
        onSettings={() => setShowKeySettings(true)}
        onLogout={handleLogout}
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
      {showKeySettings && (
        <ApiKeySettings
          current={ownKeys}
          onSave={saveOwnKeys}
          onRemove={removeOwnKeys}
          onClose={() => setShowKeySettings(false)}
        />
      )}
    </div>
  );
}

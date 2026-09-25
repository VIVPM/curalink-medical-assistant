import { useEffect, useState } from "react";

export default function Sidebar({ sessions, activeId, onSelect, onDelete, onNew, userName, credits, onLogout, onDeleteAccount, onPrivacy, onTerms }) {
  const [collapsed, setCollapsed] = useState(
    () => typeof window !== "undefined" && window.matchMedia("(max-width: 768px)").matches
  );

  useEffect(() => {
    const media = window.matchMedia("(max-width: 768px)");
    const sync = (event) => setCollapsed(event.matches);
    media.addEventListener("change", sync);
    return () => media.removeEventListener("change", sync);
  }, []);

  const closeMobileDrawer = () => {
    if (window.matchMedia("(max-width: 768px)").matches) setCollapsed(true);
  };

  const handleNew = () => {
    onNew();
    closeMobileDrawer();
  };

  const handleSelect = (id) => {
    onSelect(id);
    closeMobileDrawer();
  };

  return (
    <div className={`sidebar ${collapsed ? "sidebar-collapsed" : ""}`}>
      <div className="sidebar-header">
        {!collapsed && <span className="sidebar-logo">Curalink</span>}
        <button
          className="sidebar-toggle"
          onClick={() => setCollapsed((v) => !v)}
          title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {collapsed ? "\u203A" : "\u2039"}
        </button>
      </div>
      {!collapsed && (
        <>
          <button className="new-session-btn" onClick={handleNew}>
            + New Research Session
          </button>
          <div className="session-list">
            {sessions.map((s) => (
              <div
                key={s._id}
                className={`session-item ${s._id === activeId ? "active" : ""}`}
                onClick={() => handleSelect(s._id)}
              >
                <div className="session-copy">
                  <div className="session-title">{s.title}</div>
                  <div className="session-meta">
                    {s.messageCount || 0} messages
                  </div>
                </div>
                <button
                  className="session-delete-btn"
                  onClick={(event) => { event.stopPropagation(); onDelete(s._id); }}
                  title="Delete research session"
                  aria-label={`Delete ${s.title || "research session"}`}
                >
                  &times;
                </button>
              </div>
            ))}
            {sessions.length === 0 && (
              <div className="session-empty">No sessions yet</div>
            )}
          </div>
          <div className="sidebar-footer">
            {credits && (
              <div className="credits-badge" title="Daily questions — resets at midnight UTC">
                {credits.remaining}/{credits.cap} questions today
              </div>
            )}
            <div className="user-info">
              <span className="user-avatar">{userName?.[0]?.toUpperCase()}</span>
              <span className="user-name">{userName}</span>
            </div>
            <div className="sidebar-legal-links">
              <button onClick={onPrivacy}>Privacy</button>
              <button onClick={onTerms}>Terms</button>
            </div>
            <button className="logout-btn" onClick={onLogout}>Logout</button>
            <button className="delete-account-btn" onClick={onDeleteAccount}>Delete account</button>
          </div>
        </>
      )}
      {collapsed && (
        <div className="sidebar-collapsed-icons">
          <button className="collapsed-icon-btn" onClick={handleNew} title="New Research Session">+</button>
          <div className="sidebar-footer">
            <span className="user-avatar">{userName?.[0]?.toUpperCase()}</span>
          </div>
        </div>
      )}
    </div>
  );
}

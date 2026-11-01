// Responsive session navigation and account controls.

import { useEffect, useState } from "react";

export default function Sidebar({ sessions, activeId, onSelect, onDelete, onNew, userName, onSettings, onLogout }) {
  const [collapsed, setCollapsed] = useState(
    () => typeof window !== "undefined" && window.matchMedia("(max-width: 768px)").matches
  );

  useEffect(() => {
    const media = window.matchMedia("(max-width: 768px)");
    const sync = (event) => setCollapsed(event.matches);
    media.addEventListener("change", sync);
    return () => media.removeEventListener("change", sync);
  }, []);

  const [search, setSearch] = useState("");
  const visibleSessions = sessions.filter((session) =>
    (session.title || "").toLowerCase().includes(search.trim().toLowerCase())
  );

  const closeMobileDrawer = () => {
    if (window.matchMedia("(max-width: 768px)").matches) setCollapsed(true);
  };

  const handleNew = () => {
    setSearch("");
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
        {!collapsed && <button className="new-session-btn" onClick={handleNew}>+ New session</button>}
        <button
          className="sidebar-toggle"
          onClick={() => setCollapsed((v) => !v)}
          title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          aria-expanded={!collapsed}
        >
          {collapsed ? "\u203A" : "\u2039"}
        </button>
      </div>
      {!collapsed && (
        <>
          <div className="session-search">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" aria-hidden="true">
              <circle cx="10.5" cy="10.5" r="7" />
              <path d="m16 16 5 5" />
            </svg>
            <input type="search" aria-label="Search sessions" placeholder="Search sessions…"
              value={search} onChange={(event) => setSearch(event.target.value)} />
          </div>
          <div className="session-list">
            {visibleSessions.map((s) => (
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
            {visibleSessions.length === 0 && (
              <div className="session-empty">{search.trim() ? "No matching sessions" : "No sessions yet"}</div>
            )}
          </div>
        </>
      )}
      {collapsed && (
        <div className="sidebar-collapsed-icons">
          <button className="collapsed-icon-btn" onClick={handleNew} title="New session" aria-label="New session">+</button>
        </div>
      )}
      <div className="sidebar-footer">
        {!collapsed && <div className="sidebar-user">
          <span className="sidebar-user-dot" />
          <span className="sidebar-user-name" title={userName}>{userName}</span>
        </div>}
        <div className="sidebar-footer-actions">
          <button className="sidebar-action-btn" onClick={onSettings} title="Settings" aria-label="Settings">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="m9 3-.7 2.5-2.5 1L3.5 6l-2 3.5L3.3 11v2L1.5 14.5l2 3.5 2.3-.5 2.5 1L9 21h4l.7-2.5 2.5-1 2.3.5 2-3.5-1.8-1.5v-2l1.8-1.5-2-3.5-2.3.5-2.5-1L13 3Z" />
              <circle cx="11" cy="12" r="3" />
            </svg>
          </button>
          <button className="sidebar-action-btn sidebar-logout-btn" onClick={onLogout} title="Logout" aria-label="Logout">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4M16 17l5-5-5-5M21 12H9" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}

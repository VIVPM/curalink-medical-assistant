import { useState } from "react";

export default function Sidebar({ sessions, activeId, onSelect, onNew, userName, onLogout }) {
  const [collapsed, setCollapsed] = useState(false);

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
          <button className="new-session-btn" onClick={onNew}>
            + New Consultation
          </button>
          <div className="session-list">
            {sessions.map((s) => (
              <div
                key={s._id}
                className={`session-item ${s._id === activeId ? "active" : ""}`}
                onClick={() => onSelect(s._id)}
              >
                <div className="session-title">{s.title}</div>
                <div className="session-meta">
                  {s.messageCount || 0} messages
                </div>
              </div>
            ))}
            {sessions.length === 0 && (
              <div className="session-empty">No sessions yet</div>
            )}
          </div>
          <div className="sidebar-footer">
            <div className="user-info">
              <span className="user-avatar">{userName?.[0]?.toUpperCase()}</span>
              <span className="user-name">{userName}</span>
            </div>
            <button className="logout-btn" onClick={onLogout}>Logout</button>
          </div>
        </>
      )}
      {collapsed && (
        <div className="sidebar-collapsed-icons">
          <button className="collapsed-icon-btn" onClick={onNew} title="New Consultation">+</button>
          <div className="sidebar-footer">
            <span className="user-avatar">{userName?.[0]?.toUpperCase()}</span>
          </div>
        </div>
      )}
    </div>
  );
}

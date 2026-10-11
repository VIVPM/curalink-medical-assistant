// Audit middleware for recording successful user actions.

import { db, newId } from "../db.js";

/**
 * Audit logging middleware factory.
 * Usage: router.post("/chat", audit("chat.send", (req) => ({ type: "session", id: req.body.sessionId })), handler)
 *
 * Logs after response completes (fire-and-forget — never breaks requests).
 */
export function audit(action, getResource) {
  return (req, res, next) => {
    res.on("finish", () => {
      if (res.statusCode < 400) {
        const resource = getResource ? getResource(req) : undefined;
        db.query(
          "INSERT INTO audit_logs (id, user_id, action, resource, resource_id, metadata, ip) VALUES ($1, $2, $3, $4, $5, $6, $7)",
          [
            newId(),
            req.userId ?? null,
            action,
            resource?.type ?? null,
            resource?.id ?? null,
            { status: res.statusCode, method: req.method, path: req.originalUrl },
            req.ip,
          ]
        ).catch(() => {});
      }
    });
    next();
  };
}

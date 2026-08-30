import AuditLog from "../models/AuditLog.js";

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
        AuditLog.create({
          userId: req.userId,
          action,
          resource: resource?.type,
          resourceId: resource?.id,
          metadata: {
            status: res.statusCode,
            method: req.method,
            path: req.originalUrl,
          },
          ip: req.ip,
        }).catch(() => {}); // ponytail: fire-and-forget, audit must never break requests
      }
    });
    next();
  };
}

import { Router } from "express";
import { authMiddleware } from "../middleware/auth.js";
import { audit } from "../middleware/audit.js";
import { dispatchWebhooks } from "./webhooks.js";

const router = Router();
const FASTAPI_URL = process.env.FASTAPI_URL || "http://localhost:8000";

router.use(authMiddleware);

// POST /api/jobs — async submit (returns 202 + job_id)
router.post(
  "/",
  audit("job.submit", (req) => ({ type: "job" })),
  async (req, res) => {
    try {
      const resp = await fetch(`${FASTAPI_URL}/jobs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(req.body),
      });
      const data = await resp.json();
      res.status(resp.status).json(data);
    } catch (err) {
      res
        .status(503)
        .json({ ok: false, error: "fastapi unreachable", requestId: req.id });
    }
  }
);

// GET /api/jobs/:id — poll status
router.get("/:id", async (req, res) => {
  try {
    const resp = await fetch(`${FASTAPI_URL}/jobs/${req.params.id}`);
    const data = await resp.json();

    // Dispatch webhooks on terminal states
    if (data.state === "completed") {
      dispatchWebhooks(req.userId, "job.completed", data).catch(() => {});
    } else if (data.state === "failed") {
      dispatchWebhooks(req.userId, "job.failed", data).catch(() => {});
    }

    res.json(data);
  } catch (err) {
    res
      .status(503)
      .json({ ok: false, error: "fastapi unreachable", requestId: req.id });
  }
});

// DELETE /api/jobs/:id — cancel
router.delete(
  "/:id",
  audit("job.cancel", (req) => ({ type: "job", id: req.params.id })),
  async (req, res) => {
    try {
      const resp = await fetch(`${FASTAPI_URL}/jobs/${req.params.id}`, {
        method: "DELETE",
      });
      const data = await resp.json();
      res.json(data);
    } catch (err) {
      res
        .status(503)
        .json({ ok: false, error: "fastapi unreachable", requestId: req.id });
    }
  }
);

// GET /api/jobs/:id/events — replay SSE events (reconnect support)
router.get("/:id/events", async (req, res) => {
  const lastEventId = req.headers["last-event-id"];
  try {
    const url = new URL(`${FASTAPI_URL}/jobs/${req.params.id}/events`);
    if (lastEventId) url.searchParams.set("last_event_id", lastEventId);
    const resp = await fetch(url);
    const data = await resp.json();
    res.json(data);
  } catch (err) {
    res
      .status(503)
      .json({ ok: false, error: "fastapi unreachable", requestId: req.id });
  }
});

export default router;

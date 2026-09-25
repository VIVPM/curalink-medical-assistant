// Authenticated routes for asynchronous pipeline jobs.

import { Router } from "express";
import { authMiddleware } from "../middleware/auth.js";
import { audit } from "../middleware/audit.js";
import { dispatchWebhooks } from "./webhooks.js";

const router = Router();
const FASTAPI_URL = process.env.FASTAPI_URL || "http://localhost:8000";
const fastApiHeaders = (extra = {}) => ({
  ...extra,
  "X-Internal-API-Key": process.env.INTERNAL_API_KEY,
});

router.use(authMiddleware);


router.post(
  "/",
  audit("job.submit", (req) => ({ type: "job" })),
  async (req, res) => {
    try {
      const resp = await fetch(`${FASTAPI_URL}/jobs`, {
        method: "POST",
        headers: fastApiHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify({ ...req.body, tenant: req.userId.toString() }),
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


router.get("/:id", async (req, res) => {
  try {
    const resp = await fetch(`${FASTAPI_URL}/jobs/${req.params.id}`, {
      headers: fastApiHeaders(),
    });
    const data = await resp.json();


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


router.delete(
  "/:id",
  audit("job.cancel", (req) => ({ type: "job", id: req.params.id })),
  async (req, res) => {
    try {
      const resp = await fetch(`${FASTAPI_URL}/jobs/${req.params.id}`, {
        method: "DELETE",
        headers: fastApiHeaders(),
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


router.get("/:id/events", async (req, res) => {
  const lastEventId = req.headers["last-event-id"];
  try {
    const url = new URL(`${FASTAPI_URL}/jobs/${req.params.id}/events`);
    if (lastEventId) url.searchParams.set("last_event_id", lastEventId);
    const resp = await fetch(url, { headers: fastApiHeaders() });
    const data = await resp.json();
    res.json(data);
  } catch (err) {
    res
      .status(503)
      .json({ ok: false, error: "fastapi unreachable", requestId: req.id });
  }
});

export default router;

// Authenticated routes for asynchronous pipeline jobs.

import { Router } from "express";
import { authMiddleware } from "../middleware/auth.js";
import { audit } from "../middleware/audit.js";
import { dispatchWebhooks } from "./webhooks.js";
import { providerSettings, selectedKeys } from "../provider_settings.js";
import { ownKeysFromRequest } from "../own_keys.js";

const router = Router();
const FASTAPI_URL = process.env.FASTAPI_URL;
const fastApiHeaders = (req, extra = {}) => ({
  ...extra,
  "X-Tenant-ID": req.userId,
  "X-Internal-API-Key": process.env.INTERNAL_API_KEY,
});

router.use(authMiddleware);


router.post(
  "/",
  audit("job.submit", (req) => ({ type: "job" })),
  async (req, res) => {
    try {
      if (ownKeysFromRequest(req)) {
        return res.status(400).json({ ok: false, error: "Save credentials in account Settings before submitting background jobs." });
      }
      const settings = await providerSettings(req.userId);
      selectedKeys(settings);
      const resp = await fetch(`${FASTAPI_URL}/jobs`, {
        method: "POST",
        headers: fastApiHeaders(req, { "Content-Type": "application/json" }),
        body: JSON.stringify({ ...req.body, tenant: req.userId.toString(), credential_provider: settings.selected }),
      });
      const data = await resp.json();
      res.status(resp.status).json(data);
    } catch (err) {
      res
        .status(err.status || 503)
        .json({ ok: false, error: err.status ? err.message : "fastapi unreachable", requestId: req.id });
    }
  }
);


router.get("/:id", async (req, res) => {
  try {
    const resp = await fetch(`${FASTAPI_URL}/jobs/${req.params.id}`, {
      headers: fastApiHeaders(req),
    });
    const data = await resp.json();


    if (data.state === "completed") {
      dispatchWebhooks(req.userId, "job.completed", data).catch(() => {});
    } else if (data.state === "failed") {
      dispatchWebhooks(req.userId, "job.failed", data).catch(() => {});
    }

    res.status(resp.status).json(data);
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
        headers: fastApiHeaders(req),
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


router.get("/:id/events", async (req, res) => {
  const lastEventId = req.headers["last-event-id"];
  try {
    const url = new URL(`${FASTAPI_URL}/jobs/${req.params.id}/events`);
    if (lastEventId) url.searchParams.set("last_event_id", lastEventId);
    const resp = await fetch(url, { headers: fastApiHeaders(req) });
    const data = await resp.json();
    res.status(resp.status).json(data);
  } catch (err) {
    res
      .status(503)
      .json({ ok: false, error: "fastapi unreachable", requestId: req.id });
  }
});

export default router;

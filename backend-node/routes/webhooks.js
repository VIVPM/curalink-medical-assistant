// Authenticated webhook management and delivery.

import crypto from "crypto";
import { Router } from "express";
import { db, newId, toWebhook } from "../db.js";
import { authMiddleware } from "../middleware/auth.js";

const router = Router();
const WEBHOOK_EVENTS = ["job.completed", "job.failed"];
router.use(authMiddleware);


router.post("/", async (req, res) => {
  const { url, events } = req.body;
  if (!url) return res.status(400).json({ ok: false, error: "url required" });
  if (events !== undefined && (!Array.isArray(events) || !events.every((e) => WEBHOOK_EVENTS.includes(e)))) {
    return res.status(400).json({ ok: false, error: `events must be a subset of ${WEBHOOK_EVENTS.join(", ")}` });
  }

  const secret = crypto.randomBytes(32).toString("hex");
  const { rows } = await db.query(
    "INSERT INTO webhooks (id, user_id, url, secret, events) VALUES ($1, $2, $3, $4, $5) RETURNING *",
    [newId(), req.userId, url, secret, events || ["job.completed"]]
  );
  const wh = toWebhook(rows[0]);

  res.status(201).json({
    ok: true,
    webhook: { id: wh._id, url: wh.url, secret, events: wh.events },
  });
});


router.get("/", async (req, res) => {
  const { rows } = await db.query("SELECT * FROM webhooks WHERE user_id = $1", [req.userId]);
  res.json({ ok: true, webhooks: rows.map(toWebhook) });
});


router.delete("/:id", async (req, res) => {
  await db.query("DELETE FROM webhooks WHERE id = $1 AND user_id = $2", [req.params.id, req.userId]);
  res.json({ ok: true, deleted: true });
});

/**
 * Dispatch webhook notifications for a user event.
 * Fire-and-forget — webhook delivery must never block the main flow.
 */
export async function dispatchWebhooks(userId, event, payload) {
  try {
    const { rows: hooks } = await db.query(
      "SELECT * FROM webhooks WHERE user_id = $1 AND active AND $2 = ANY(events)",
      [userId, event]
    );

    for (const hook of hooks) {
      try {
        const body = JSON.stringify({
          event,
          payload,
          timestamp: new Date().toISOString(),
        });
        const signature = crypto
          .createHmac("sha256", hook.secret)
          .update(body)
          .digest("hex");


        fetch(hook.url, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-Webhook-Signature": signature,
          },
          body,
          signal: AbortSignal.timeout(10_000),
        }).catch(() => {});
      } catch {}
    }
  } catch {}
}

export default router;

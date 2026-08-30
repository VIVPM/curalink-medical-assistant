import crypto from "crypto";
import { Router } from "express";
import Webhook from "../models/Webhook.js";
import { authMiddleware } from "../middleware/auth.js";

const router = Router();
router.use(authMiddleware);

// POST /api/webhooks — register a webhook
router.post("/", async (req, res) => {
  const { url, events } = req.body;
  if (!url) return res.status(400).json({ ok: false, error: "url required" });

  const secret = crypto.randomBytes(32).toString("hex");
  const wh = await Webhook.create({
    userId: req.userId,
    url,
    secret,
    events: events || ["job.completed"],
  });

  res.status(201).json({
    ok: true,
    webhook: { id: wh._id, url: wh.url, secret, events: wh.events },
  });
});

// GET /api/webhooks — list user's webhooks
router.get("/", async (req, res) => {
  const hooks = await Webhook.find({ userId: req.userId })
    .select("-secret")
    .lean();
  res.json({ ok: true, webhooks: hooks });
});

// DELETE /api/webhooks/:id — remove a webhook
router.delete("/:id", async (req, res) => {
  await Webhook.findOneAndDelete({ _id: req.params.id, userId: req.userId });
  res.json({ ok: true, deleted: true });
});

/**
 * Dispatch webhook notifications for a user event.
 * Fire-and-forget — webhook delivery must never block the main flow.
 */
export async function dispatchWebhooks(userId, event, payload) {
  try {
    const hooks = await Webhook.find({
      userId,
      active: true,
      events: event,
    }).lean();

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

        // Fire-and-forget with 10s timeout
        fetch(hook.url, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-Webhook-Signature": signature,
          },
          body,
          signal: AbortSignal.timeout(10_000),
        }).catch(() => {});
      } catch {
        // individual hook failure — skip, try next
      }
    }
  } catch {
    // query failure — swallow, webhooks are best-effort
  }
}

export default router;

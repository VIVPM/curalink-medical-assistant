// Authenticated research-session CRUD routes.

import { Router } from "express";
import rateLimit, { ipKeyGenerator } from "express-rate-limit";
import { db, newId, toMessage, toSession } from "../db.js";
import { authMiddleware } from "../middleware/auth.js";

const router = Router();


const createSessionLimiter = rateLimit({
  windowMs: 60 * 60 * 1000,
  max: Number(process.env.SESSION_RATE_MAX) || 30,
  standardHeaders: true,
  legacyHeaders: false,
  keyGenerator: (req) => req.userId || ipKeyGenerator(req.ip),
  message: { ok: false, error: "too many sessions created, try again later" },
});


router.use(authMiddleware);


router.post("/session", createSessionLimiter, async (req, res) => {
  const { disease, intent, location } = req.body;

  if (!disease || !disease.trim()) {
    return res.status(400).json({ ok: false, error: "disease is required" });
  }

  const d = disease.trim();
  const i = (intent || "").trim();
  const { rows } = await db.query(
    `INSERT INTO sessions (id, user_id, disease, intent, location, title)
     VALUES ($1, $2, $3, $4, $5, $6) RETURNING *`,
    [newId(), req.userId, d, i, (location || "").trim(), i ? `${d} — ${i}` : d]
  );

  res.status(201).json({ ok: true, session: toSession(rows[0]) });
});


router.get("/sessions", async (req, res) => {
  const { rows } = await db.query(
    `SELECT id, title, disease, message_count, created_at FROM sessions
     WHERE user_id = $1 ORDER BY created_at DESC LIMIT 50`,
    [req.userId]
  );
  const sessions = rows.map((r) => ({
    _id: r.id,
    title: r.title,
    staticContext: { disease: r.disease },
    messageCount: r.message_count,
    createdAt: r.created_at,
  }));

  res.json({ ok: true, sessions });
});


router.get("/session/:id", async (req, res) => {
  const found = await db.query("SELECT * FROM sessions WHERE id = $1 AND user_id = $2", [req.params.id, req.userId]);
  const session = toSession(found.rows[0]);

  if (!session) {
    return res.status(404).json({ ok: false, error: "session not found" });
  }

  const { rows } = await db.query("SELECT * FROM messages WHERE session_id = $1 ORDER BY created_at", [session._id]);

  res.json({ ok: true, session, messages: rows.map(toMessage) });
});


router.delete("/session/:id", async (req, res) => {
  const { rowCount } = await db.query("DELETE FROM sessions WHERE id = $1 AND user_id = $2", [req.params.id, req.userId]);

  if (!rowCount) {
    return res.status(404).json({ ok: false, error: "session not found" });
  }

  res.json({ ok: true, deleted: true });
});

export default router;

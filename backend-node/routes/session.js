import { Router } from "express";
import Session from "../models/Session.js";
import Message from "../models/Message.js";
import { authMiddleware } from "../middleware/auth.js";

const router = Router();

// All session routes require auth
router.use(authMiddleware);

// POST /api/session — create new session (locks form)
router.post("/session", async (req, res) => {
  const { disease, intent, location, patientName } = req.body;

  if (!disease || !disease.trim()) {
    return res.status(400).json({ ok: false, error: "disease is required" });
  }

  const session = await Session.create({
    userId: req.userId,
    staticContext: {
      disease: disease.trim(),
      intent: (intent || "").trim(),
      location: (location || "").trim(),
      patientName: (patientName || "").trim(),
    },
  });

  res.status(201).json({ ok: true, session });
});

// GET /api/sessions — list sessions for current user (sidebar)
router.get("/sessions", async (req, res) => {
  const sessions = await Session.find({ userId: req.userId })
    .select("title staticContext.disease messageCount createdAt")
    .sort({ createdAt: -1 })
    .limit(50)
    .lean();

  res.json({ ok: true, sessions });
});

// GET /api/session/:id — fetch session + messages (only if owned by user)
router.get("/session/:id", async (req, res) => {
  const session = await Session.findOne({
    _id: req.params.id,
    userId: req.userId,
  }).lean();

  if (!session) {
    return res.status(404).json({ ok: false, error: "session not found" });
  }

  const messages = await Message.find({ sessionId: session._id })
    .sort({ createdAt: 1 })
    .lean();

  res.json({ ok: true, session, messages });
});

// DELETE /api/session/:id — delete session + its messages
router.delete("/session/:id", async (req, res) => {
  const session = await Session.findOne({
    _id: req.params.id,
    userId: req.userId,
  });

  if (!session) {
    return res.status(404).json({ ok: false, error: "session not found" });
  }

  await Message.deleteMany({ sessionId: session._id });
  await session.deleteOne();

  res.json({ ok: true, deleted: true });
});

export default router;

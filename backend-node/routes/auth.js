// Authentication routes for signup, login, and current-user lookup.

import { Router } from "express";
import rateLimit from "express-rate-limit";
import bcrypt from "bcryptjs";
import { db, newId, toUser } from "../db.js";
import { signToken, signRefreshToken, verifyRefreshToken, authMiddleware } from "../middleware/auth.js";

const router = Router();

const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: Number(process.env.AUTH_RATE_MAX) || 20,
  standardHeaders: true,
  legacyHeaders: false,
  message: { ok: false, error: "too many attempts, try again later" },
});


router.post("/signup", authLimiter, async (req, res) => {
  const { name, email, password, acceptTerms } = req.body;

  if (!name || !email || !password) {
    return res.status(400).json({ ok: false, error: "name, email, and password required" });
  }
  if (password.length < 8) {
    return res.status(400).json({ ok: false, error: "password must be at least 8 characters" });
  }
  if (acceptTerms !== true) {
    return res.status(400).json({ ok: false, error: "terms and privacy notice must be accepted" });
  }

  const { rows } = await db.query(
    `INSERT INTO users (id, name, email, password, terms_accepted_at, terms_version, credits)
     VALUES ($1, $2, $3, $4, now(), $5, $6) ON CONFLICT (email) DO NOTHING RETURNING *`,
    [
      newId(),
      name.trim(),
      email.trim().toLowerCase(),
      await bcrypt.hash(password, 10),
      "2026-09-25",
      Number(process.env.DAILY_MESSAGE_CAP) || 5,
    ]
  );
  if (!rows[0]) {
    return res.status(409).json({ ok: false, error: "email already registered" });
  }

  const user = toUser(rows[0]);
  const token = signToken(user._id, user.authVersion);

  res.status(201).json({
    ok: true,
    token,
    refreshToken: signRefreshToken(user._id, user.authVersion),
    user: { _id: user._id, name: user.name, email: user.email },
  });
});


router.post("/login", authLimiter, async (req, res) => {
  const { email, password } = req.body;

  if (!email || !password) {
    return res.status(400).json({ ok: false, error: "email and password required" });
  }

  const { rows } = await db.query("SELECT * FROM users WHERE email = $1", [email.trim().toLowerCase()]);
  if (!rows[0] || !(await bcrypt.compare(password, rows[0].password))) {
    return res.status(401).json({ ok: false, error: "invalid credentials" });
  }

  const user = toUser(rows[0]);
  const token = signToken(user._id, user.authVersion);

  res.json({
    ok: true,
    token,
    refreshToken: signRefreshToken(user._id, user.authVersion),
    user: { _id: user._id, name: user.name, email: user.email },
  });
});


router.post("/refresh", async (req, res) => {
  try {
    const decoded = verifyRefreshToken(req.body.refreshToken);
    const { rows } = await db.query("SELECT * FROM users WHERE id = $1", [decoded.userId]);
    const user = toUser(rows[0]);
    if (!user || user.authVersion !== decoded.authVersion) {
      return res.status(401).json({ ok: false, error: "session expired" });
    }
    res.json({
      ok: true,
      token: signToken(user._id, user.authVersion),
      refreshToken: signRefreshToken(user._id, user.authVersion),
    });
  } catch {
    res.status(401).json({ ok: false, error: "session expired" });
  }
});

router.post("/logout", authMiddleware, async (req, res) => {
  await db.query("UPDATE users SET auth_version = auth_version + 1, updated_at = now() WHERE id = $1", [req.userId]);
  res.json({ ok: true });
});

router.get("/me", authMiddleware, async (req, res) => {
  const { rows } = await db.query("SELECT * FROM users WHERE id = $1", [req.userId]);
  const user = toUser(rows[0]);
  if (!user) return res.status(401).json({ ok: false, error: "user not found" });
  res.json({ ok: true, user });
});

export default router;

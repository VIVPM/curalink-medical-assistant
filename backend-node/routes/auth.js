// Authentication routes for signup, login, and current-user lookup.

import { Router } from "express";
import rateLimit from "express-rate-limit";
import User from "../models/User.js";
import { signToken, signRefreshToken, verifyRefreshToken, authMiddleware } from "../middleware/auth.js";

const router = Router();

// Brute-force limit applies to credential endpoints only, so session checks
// on page load (GET /me) never exhaust it and log users out.
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

  const existing = await User.findOne({ email: email.toLowerCase() });
  if (existing) {
    return res.status(409).json({ ok: false, error: "email already registered" });
  }

  const user = await User.create({
    name,
    email,
    password,
    termsAcceptedAt: new Date(),
    termsVersion: "2026-09-25",
  });
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

  const user = await User.findOne({ email: email.toLowerCase() });
  if (!user) {
    return res.status(401).json({ ok: false, error: "invalid credentials" });
  }

  const match = await user.comparePassword(password);
  if (!match) {
    return res.status(401).json({ ok: false, error: "invalid credentials" });
  }

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
    const user = await User.findById(decoded.userId);
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
  await User.updateOne({ _id: req.userId }, { $inc: { authVersion: 1 } });
  res.json({ ok: true });
});

router.get("/me", authMiddleware, async (req, res) => {
  const user = await User.findById(req.userId).select("-password");
  if (!user) return res.status(401).json({ ok: false, error: "user not found" });
  res.json({ ok: true, user });
});

export default router;

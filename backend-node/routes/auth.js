import { Router } from "express";
import User from "../models/User.js";
import { signToken } from "../middleware/auth.js";

const router = Router();

// POST /api/auth/signup
router.post("/signup", async (req, res) => {
  const { name, email, password } = req.body;

  if (!name || !email || !password) {
    return res.status(400).json({ ok: false, error: "name, email, and password required" });
  }
  if (password.length < 6) {
    return res.status(400).json({ ok: false, error: "password must be at least 6 characters" });
  }

  const existing = await User.findOne({ email: email.toLowerCase() });
  if (existing) {
    return res.status(409).json({ ok: false, error: "email already registered" });
  }

  const user = await User.create({ name, email, password });
  const token = signToken(user._id);

  res.status(201).json({
    ok: true,
    token,
    user: { _id: user._id, name: user.name, email: user.email },
  });
});

// POST /api/auth/login
router.post("/login", async (req, res) => {
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

  const token = signToken(user._id);

  res.json({
    ok: true,
    token,
    user: { _id: user._id, name: user.name, email: user.email },
  });
});

// GET /api/auth/me — verify token + get user info
router.get("/me", async (req, res) => {
  const header = req.headers.authorization;
  if (!header || !header.startsWith("Bearer ")) {
    return res.status(401).json({ ok: false, error: "not authenticated" });
  }

  try {
    const jwt = await import("jsonwebtoken");
    const decoded = jwt.default.verify(
      header.slice(7),
      process.env.JWT_SECRET || "curalink-dev-secret"
    );
    const user = await User.findById(decoded.userId).select("-password");
    if (!user) return res.status(401).json({ ok: false, error: "user not found" });
    res.json({ ok: true, user });
  } catch {
    return res.status(401).json({ ok: false, error: "invalid token" });
  }
});

export default router;

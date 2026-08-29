import crypto from "crypto";
import express from "express";
import cors from "cors";
import rateLimit from "express-rate-limit";
import dotenv from "dotenv";
import mongoose from "mongoose";
import authRouter from "./routes/auth.js";
import sessionRouter from "./routes/session.js";
import chatRouter from "./routes/chat.js";
import { redisStatus } from "./cache.js";
import { authMiddleware } from "./middleware/auth.js";
import Session from "./models/Session.js";
import Message from "./models/Message.js";

dotenv.config();

const app = express();
// Behind Render's proxy: trust the first hop so req.ip is the real client IP,
// which the rate limiters key on (SEC-4). One proxy hop on Render.
app.set("trust proxy", 1);
const PORT = process.env.PORT || 4000;
const FASTAPI_URL = process.env.FASTAPI_URL || "http://localhost:8000";
const MONGO_URI = process.env.MONGO_URI;

if (!MONGO_URI) {
  console.error("MONGO_URI not set in .env");
  process.exit(1);
}

// Fail fast on missing secrets rather than falling back to insecure defaults.
if (!process.env.JWT_SECRET) {
  console.error("JWT_SECRET not set in .env"); // SEC-2
  process.exit(1);
}

mongoose
  .connect(MONGO_URI)
  .then(() => {
    console.log("MongoDB connected");
  })
  .catch((err) => {
    console.error("MongoDB connection failed:", err.message);
    process.exit(1);
  });

// Per-request id + structured access log (REL-1). Lightweight — one JSON line
// per request; no logging dependency for what a few lines do.
app.use((req, res, next) => {
  req.id = crypto.randomUUID();
  const start = Date.now();
  res.on("finish", () => {
    console.log(
      JSON.stringify({
        t: new Date().toISOString(),
        id: req.id,
        method: req.method,
        url: req.originalUrl,
        status: res.statusCode,
        ms: Date.now() - start,
      })
    );
  });
  next();
});

app.use(express.json({ limit: "16kb" })); // body cap (SEC-6)

// CORS allow-list from env (comma-separated); defaults to deployed frontend + localhost.
const _DEFAULT_ORIGINS =
  "http://localhost:5173,https://curalink-medical-assistant-frontend.onrender.com";
const ALLOWED_ORIGINS = (process.env.ALLOWED_ORIGINS || _DEFAULT_ORIGINS)
  .split(",")
  .map((o) => o.trim())
  .filter(Boolean);
app.use(cors({ origin: ALLOWED_ORIGINS, credentials: true }));

app.get("/", (req, res) => {
  res.json({ ok: true, service: "curalink-express", version: "1.0.0" });
});

app.get("/health", (req, res) => {
  res.json({ ok: true, service: "express", redis: redisStatus() });
});

// Strict rate limit on auth to blunt brute-force / signup spam (SEC-4).
const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: Number(process.env.AUTH_RATE_MAX) || 20,
  standardHeaders: true,
  legacyHeaders: false,
  message: { ok: false, error: "too many attempts, try again later" },
});

// Routes
app.use("/api/auth", authLimiter, authRouter);
app.use("/api", sessionRouter);
app.use("/api", chatRouter);

// Daily credits: 1 credit = 1 question, auto-resets at UTC midnight.
const DAILY_MESSAGE_CAP = Number(process.env.DAILY_MESSAGE_CAP) || 5;

app.get("/api/account/credits", authMiddleware, async (req, res) => {
  const since = new Date();
  since.setUTCHours(0, 0, 0, 0);
  const sessionIds = await Session.find({ userId: req.userId }).distinct("_id");
  const used = sessionIds.length
    ? await Message.countDocuments({
        sessionId: { $in: sessionIds },
        role: "user",
        createdAt: { $gte: since },
      })
    : 0;
  res.json({ ok: true, cap: DAILY_MESSAGE_CAP, used, remaining: Math.max(0, DAILY_MESSAGE_CAP - used) });
});

app.get("/api/ping", async (req, res) => {
  try {
    const response = await fetch(`${FASTAPI_URL}/health`);
    if (!response.ok) {
      return res.status(502).json({
        ok: false,
        error: `fastapi responded with ${response.status}`,
      });
    }
    const data = await response.json();
    res.json({ ok: true, fastapi: data });
  } catch (err) {
    console.error(JSON.stringify({ id: req.id, level: "error", where: "api_ping", error: err.message }));
    res.status(503).json({ ok: false, error: "fastapi unreachable", requestId: req.id });
  }
});

// Central error handler (REL-1). Logs the error with its request id and returns
// a generic message + id — never a stack trace to the client (also helps SEC-7).
// Express 5 forwards rejected async route handlers here too.
app.use((err, req, res, next) => {
  console.error(
    JSON.stringify({
      t: new Date().toISOString(),
      id: req.id,
      level: "error",
      method: req.method,
      url: req.originalUrl,
      error: err.message,
      stack: err.stack,
    })
  );
  if (res.headersSent) return next(err);
  res.status(err.status || 500).json({
    ok: false,
    error: "internal server error",
    requestId: req.id,
  });
});

const server = app.listen(PORT, () => {
  console.log(`Express listening on http://localhost:${PORT}`);
});

// Graceful shutdown: stop accepting, let in-flight requests drain, then exit.
function shutdown(signal) {
  console.log(`[shutdown] ${signal} received — draining connections...`);
  server.close(() => {
    console.log("[shutdown] HTTP server closed");
    mongoose.connection.close(false).then(() => {
      console.log("[shutdown] MongoDB disconnected");
      process.exit(0);
    });
  });
  // Force exit after 30s if drain stalls (above p99 pipeline duration)
  setTimeout(() => {
    console.error("[shutdown] forced exit after 30s timeout");
    process.exit(1);
  }, 30_000).unref();
}
process.on("SIGTERM", () => shutdown("SIGTERM"));
process.on("SIGINT", () => shutdown("SIGINT"));

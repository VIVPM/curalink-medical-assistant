// Express API entrypoint for authentication, sessions, chat, and account operations.

import "dotenv/config";
import crypto from "crypto";
import express from "express";
import cors from "cors";
import authRouter from "./routes/auth.js";
import sessionRouter from "./routes/session.js";
import chatRouter from "./routes/chat.js";
import jobsRouter from "./routes/jobs.js";
import webhooksRouter from "./routes/webhooks.js";
import { cacheDeleteUser, redisStatus } from "./cache.js";
import { authMiddleware } from "./middleware/auth.js";
import { validateOwnKeys } from "./own_keys.js";
import { closeDb, db, initDb } from "./db.js";


const app = express();


app.set("trust proxy", 1);
const PORT = process.env.PORT || 4000;
const FASTAPI_URL = process.env.FASTAPI_URL;
for (const name of ["FASTAPI_URL", "ALLOWED_ORIGINS"]) {
  if (!process.env[name]) {
    console.error(`${name} not set in .env`);
    process.exit(1);
  }
}
if (!process.env.DATABASE_URL) {
  console.error("DATABASE_URL not set in .env");
  process.exit(1);
}


if (!process.env.JWT_SECRET) {
  console.error("JWT_SECRET not set in .env");
  process.exit(1);
}
if (!process.env.INTERNAL_API_KEY) {
  console.error("INTERNAL_API_KEY not set in .env");
  process.exit(1);
}

initDb()
  .then(() => console.log("Postgres connected"))
  .catch((err) => {
    console.error("Postgres connection failed:", err.message);
    process.exit(1);
  });



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

app.use(express.json({ limit: "16kb" }));


const ALLOWED_ORIGINS = process.env.ALLOWED_ORIGINS
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

app.use("/api/auth", authRouter);
app.use("/api", sessionRouter);
app.use("/api", chatRouter);
app.use("/api/jobs", jobsRouter);
app.use("/api/webhooks", webhooksRouter);


const DAILY_MESSAGE_CAP = Number(process.env.DAILY_MESSAGE_CAP) || 5;

app.get("/api/account/keys/provider", authMiddleware, async (req, res) => {
  try {
    const response = await fetch(`${FASTAPI_URL}/keys/provider`, {
      headers: { "X-Internal-API-Key": process.env.INTERNAL_API_KEY },
    });
    if (!response.ok) return res.status(503).json({ ok: false, error: "Provider unavailable" });
    res.json(await response.json());
  } catch {
    res.status(503).json({ ok: false, error: "Provider unavailable" });
  }
});

app.post("/api/account/keys/validate", authMiddleware, async (req, res) => {
  try {
    await validateOwnKeys(req.body);
    res.json({ ok: true });
  } catch (error) {
    res.status(error.status || 503).json({ ok: false, error: error.message });
  }
});

app.get("/api/account/credits", authMiddleware, async (req, res) => {
  const since = new Date();
  since.setUTCHours(0, 0, 0, 0);
  const { rows } = await db.query(
    `SELECT count(*)::int AS n FROM messages m JOIN sessions s ON s.id = m.session_id
     WHERE s.user_id = $1 AND m.role = 'user' AND NOT m.own_key AND m.created_at >= $2`,
    [req.userId, since]
  );
  const used = rows[0].n;
  res.json({ ok: true, cap: DAILY_MESSAGE_CAP, used, remaining: Math.max(0, DAILY_MESSAGE_CAP - used) });
});



app.delete("/api/account", authMiddleware, async (req, res) => {
  await cacheDeleteUser(req.userId);
  await db.query("DELETE FROM users WHERE id = $1", [req.userId]);
  res.json({ ok: true, deleted: true });
});

app.get("/api/ping", async (req, res) => {
  try {
    const response = await fetch(`${FASTAPI_URL}/health`, {
      headers: { "X-Internal-API-Key": process.env.INTERNAL_API_KEY },
    });
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
    closeDb().then(() => {
      console.log("[shutdown] Postgres disconnected");
      process.exit(0);
    });
  });

  setTimeout(() => {
    console.error("[shutdown] forced exit after 30s timeout");
    process.exit(1);
  }, 30_000).unref();
}
process.on("SIGTERM", () => shutdown("SIGTERM"));
process.on("SIGINT", () => shutdown("SIGINT"));

import crypto from "crypto";
import { Router } from "express";
import rateLimit, { ipKeyGenerator } from "express-rate-limit";
import Session from "../models/Session.js";
import Message from "../models/Message.js";
import User from "../models/User.js";
import { cacheGet, cacheSet } from "../cache.js";
import { authMiddleware } from "../middleware/auth.js";

const router = Router();

const FASTAPI_URL = process.env.FASTAPI_URL || "http://localhost:8000";  // overridden by env at deploy
const CACHE_TTL_MS = 24 * 60 * 60 * 1000;
const MAX_MESSAGE_LENGTH = 4000; // cap the expensive path (SEC-6)
const DAILY_MESSAGE_CAP = Number(process.env.DAILY_MESSAGE_CAP) || 5;
// Token-aware daily cap: if set, limits total tokens/day instead of just messages.
// Rough estimate: 1 token ≈ 4 chars. 0 = disabled (message-count only).
const DAILY_TOKEN_CAP = Number(process.env.DAILY_TOKEN_CAP) || 0;

// Idempotency: in-memory store with 5-min TTL. Prevents duplicate pipeline runs
// when the client retries on a timeout. Keyed on userId + Idempotency-Key header.
// ponytail: in-memory is fine for single instance; move to Redis with SCALE-8.
const _idempotencyStore = new Map();
const IDEMPOTENCY_TTL_MS = 5 * 60 * 1000;

function idempotencyCheck(userId, key) {
  const k = `${userId}:${key}`;
  const entry = _idempotencyStore.get(k);
  if (entry && Date.now() - entry.ts < IDEMPOTENCY_TTL_MS) return entry.res;
  return null;
}

function idempotencySet(userId, key, res) {
  const k = `${userId}:${key}`;
  _idempotencyStore.set(k, { ts: Date.now(), res });
  // Lazy eviction: drop expired entries when store gets large
  if (_idempotencyStore.size > 10_000) {
    const now = Date.now();
    for (const [mk, mv] of _idempotencyStore) {
      if (now - mv.ts > IDEMPOTENCY_TTL_MS) _idempotencyStore.delete(mk);
    }
  }
}

// Count user-role messages sent today (since UTC midnight) across all of a
// user's sessions. This IS the credit mechanism — remaining = cap - used.
// At midnight the window moves and the count is 0 again; no column to
// decrement, no nightly restore job. (Pattern from multi-crew-lead-coordinator.)
async function messagesUsedToday(userId) {
  const since = new Date();
  since.setUTCHours(0, 0, 0, 0);
  const sessionIds = await Session.find({ userId }).distinct("_id");
  if (!sessionIds.length) return 0;
  return Message.countDocuments({
    sessionId: { $in: sessionIds },
    role: "user",
    createdAt: { $gte: since },
  });
}

// Token-aware daily usage: sum estimated tokens from today's messages.
// Rough: 1 token ≈ 4 chars. Returns 0 if DAILY_TOKEN_CAP is disabled.
async function tokensUsedToday(userId) {
  if (!DAILY_TOKEN_CAP) return 0;
  const since = new Date();
  since.setUTCHours(0, 0, 0, 0);
  const sessionIds = await Session.find({ userId }).distinct("_id");
  if (!sessionIds.length) return 0;
  const msgs = await Message.find({
    sessionId: { $in: sessionIds },
    createdAt: { $gte: since },
  })
    .select("content")
    .lean();
  return msgs.reduce((sum, m) => sum + Math.ceil((m.content || "").length / 4), 0);
}

// Per-user quota on the expensive pipeline (SEC-4). Keyed by user id (set by
// authMiddleware) so one account can't spam costly LLM/retrieval runs.
const chatLimiter = rateLimit({
  windowMs: 60 * 1000,
  max: Number(process.env.CHAT_RATE_MAX) || 15,
  standardHeaders: true,   // sends RateLimit-* + Retry-After headers
  legacyHeaders: false,
  keyGenerator: (req) => req.userId || ipKeyGenerator(req.ip),
  message: { ok: false, error: "rate limit exceeded, please slow down" },
});

// Conservative normalization: case, possessives, punctuation, whitespace. Lets
// trivial re-typings share a cache entry (UX-3) — "Parkinson's?" == "parkinsons".
// Deliberately does NOT strip stopwords (would collapse distinct questions into
// one wrong answer) and does NOT resolve synonyms/abbreviations like "DBS" vs
// "deep brain stimulation" — that needs a semantic cache, deferred to SCALE-1.
function normKey(s) {
  return (s || "")
    .toLowerCase()
    .replace(/['‘’]/g, "")   // possessive: parkinson's -> parkinsons
    .replace(/[^\w\s]/g, " ")           // other punctuation -> space
    .replace(/\s+/g, " ")
    .trim();
}

function cacheKey(disease, intent, message, history = []) {
  const normalized = `${normKey(disease)}|${normKey(intent)}|${normKey(message)}`;
  // The answer depends on prior conversation, so the key must too. Without this,
  // the same follow-up text in two different chats collides on one cached answer
  // (BUG-1). First turns have empty history, so cross-session hits still work.
  const historyStr = history
    .map((m) => `${m.role}:${normKey(m.content)}`)
    .join("|");
  return crypto
    .createHash("sha256")
    .update(`${normalized}||${historyStr}`)
    .digest("hex");
}

// All chat routes require auth, then a per-user rate limit
router.use(authMiddleware);
router.use(chatLimiter);

// POST /api/chat — send message, run pipeline, return structured response
router.post("/chat", async (req, res) => {
  const { sessionId, message } = req.body;

  if (!sessionId) {
    return res.status(400).json({ ok: false, error: "sessionId is required" });
  }
  if (!message || !message.trim()) {
    return res.status(400).json({ ok: false, error: "message is required" });
  }
  if (message.length > MAX_MESSAGE_LENGTH) {
    return res.status(400).json({ ok: false, error: "message too long" });
  }

  // Idempotency: if the client sends the same key within 5 min, return the
  // previous response instead of re-running the pipeline.
  const idempKey = req.headers["idempotency-key"];
  if (idempKey) {
    const prev = idempotencyCheck(req.userId, idempKey);
    if (prev) return res.json(prev);
  }

  // 1. Load session
  const session = await Session.findOne({ _id: sessionId, userId: req.userId });
  if (!session) {
    return res.status(404).json({ ok: false, error: "session not found" });
  }

  // Daily quota: 1 credit = 1 question. Window-based — count today's messages,
  // auto-resets at UTC midnight. No decrement, no nightly job.
  const used = await messagesUsedToday(req.userId);
  if (used >= DAILY_MESSAGE_CAP) {
    // Retry-After: seconds until next UTC midnight
    const now = new Date();
    const midnight = new Date(now);
    midnight.setUTCDate(midnight.getUTCDate() + 1);
    midnight.setUTCHours(0, 0, 0, 0);
    const retryAfter = Math.ceil((midnight - now) / 1000);
    res.setHeader("Retry-After", retryAfter);
    return res
      .status(402)
      .json({ ok: false, error: `Daily limit reached (${DAILY_MESSAGE_CAP} questions/day). Resets at midnight UTC.` });
  }

  // Token-aware daily cap: count tokens consumed, not just messages.
  if (DAILY_TOKEN_CAP) {
    const tUsed = await tokensUsedToday(req.userId);
    if (tUsed >= DAILY_TOKEN_CAP) {
      const now = new Date();
      const midnight = new Date(now);
      midnight.setUTCDate(midnight.getUTCDate() + 1);
      midnight.setUTCHours(0, 0, 0, 0);
      res.setHeader("Retry-After", Math.ceil((midnight - now) / 1000));
      return res.status(402).json({
        ok: false,
        error: `Daily token limit reached (${tUsed}/${DAILY_TOKEN_CAP} tokens). Resets at midnight UTC.`,
      });
    }
  }

  // 2. Load chat history
  const history = await Message.find({ sessionId })
    .sort({ createdAt: 1 })
    .select("role content")
    .lean();

  const recentMessages = history.map((m) => ({
    role: m.role,
    content: m.content,
  }));

  // 3. Save user message FIRST (survives pipeline crashes)
  const userMsg = await Message.create({
    sessionId,
    role: "user",
    content: message.trim(),
  });

  // Query-result cache check
  const ckey = cacheKey(
    session.staticContext.disease,
    session.staticContext.intent,
    message,
    recentMessages
  );
  const cachedResponse = await cacheGet(ckey);
  if (cachedResponse) {
    const assistantMsg = await Message.create({
      sessionId,
      role: "assistant",
      content: cachedResponse.overview || JSON.stringify(cachedResponse),
      structuredResponse: cachedResponse,
      pipelineMeta: cachedResponse.pipelineMeta || null,
    });
    await Session.findByIdAndUpdate(sessionId, { $inc: { messageCount: 2 } });
    return res.json({
      ok: true,
      userMessage: userMsg,
      assistantMessage: assistantMsg,
      response: cachedResponse,
      cached: true,
    });
  }

  // 4. Call FastAPI /pipeline/run
  const pipelineBody = {
    static: {
      disease: session.staticContext.disease,
      intent: session.staticContext.intent,
      location: session.staticContext.location,
      patientName: session.staticContext.patientName,
    },
    dynamic: {
      recentMessages,
    },
    current: {
      userMessage: message.trim(),
    },
  };

  let pipelineResult;
  try {
    const resp = await fetch(`${FASTAPI_URL}/pipeline/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(pipelineBody),
    });

    if (!resp.ok) {
      const detail = await resp.text();
      console.error(JSON.stringify({ id: req.id, level: "error", where: "pipeline_run", status: resp.status, detail }));
      return res.status(502).json({ ok: false, error: "pipeline failed", requestId: req.id });
    }

    pipelineResult = await resp.json();
  } catch (err) {
    console.error(JSON.stringify({ id: req.id, level: "error", where: "pipeline_run", error: err.message }));
    return res.status(503).json({ ok: false, error: "fastapi unreachable", requestId: req.id });
  }

  // 5. Save assistant message + pipeline meta
  const assistantContent =
    pipelineResult.overview || JSON.stringify(pipelineResult);

  const assistantMsg = await Message.create({
    sessionId,
    role: "assistant",
    content: assistantContent,
    structuredResponse: pipelineResult,
    pipelineMeta: pipelineResult.pipelineMeta || null,
  });

  // 6. Update session message count
  await Session.findByIdAndUpdate(sessionId, {
    $inc: { messageCount: 2 },
  });

  // 7. Cache the result (skip abstain responses — they signal no useful info)
  if (!pipelineResult.abstain_reason) {
    await cacheSet(ckey, pipelineResult, CACHE_TTL_MS);
  }

  // 8. Return response
  const result = {
    ok: true,
    userMessage: userMsg,
    assistantMessage: assistantMsg,
    response: pipelineResult,
  };
  if (idempKey) idempotencySet(req.userId, idempKey, result);
  res.json(result);
});

// POST /api/chat/stream — SSE streaming version
router.post("/chat/stream", async (req, res) => {
  const { sessionId, message } = req.body;

  if (!sessionId) {
    return res.status(400).json({ ok: false, error: "sessionId is required" });
  }
  if (!message || !message.trim()) {
    return res.status(400).json({ ok: false, error: "message is required" });
  }
  if (message.length > MAX_MESSAGE_LENGTH) {
    return res.status(400).json({ ok: false, error: "message too long" });
  }

  const session = await Session.findOne({ _id: sessionId, userId: req.userId });
  if (!session) {
    return res.status(404).json({ ok: false, error: "session not found" });
  }

  // Daily quota check (same window-based logic as /chat)
  const used = await messagesUsedToday(req.userId);
  if (used >= DAILY_MESSAGE_CAP) {
    const now = new Date();
    const midnight = new Date(now);
    midnight.setUTCDate(midnight.getUTCDate() + 1);
    midnight.setUTCHours(0, 0, 0, 0);
    res.setHeader("Retry-After", Math.ceil((midnight - now) / 1000));
    return res
      .status(402)
      .json({ ok: false, error: `Daily limit reached (${DAILY_MESSAGE_CAP} questions/day). Resets at midnight UTC.` });
  }

  // Token-aware daily cap (same as /chat)
  if (DAILY_TOKEN_CAP) {
    const tUsed = await tokensUsedToday(req.userId);
    if (tUsed >= DAILY_TOKEN_CAP) {
      const now = new Date();
      const midnight = new Date(now);
      midnight.setUTCDate(midnight.getUTCDate() + 1);
      midnight.setUTCHours(0, 0, 0, 0);
      res.setHeader("Retry-After", Math.ceil((midnight - now) / 1000));
      return res.status(402).json({
        ok: false,
        error: `Daily token limit reached (${tUsed}/${DAILY_TOKEN_CAP} tokens). Resets at midnight UTC.`,
      });
    }
  }

  const history = await Message.find({ sessionId })
    .sort({ createdAt: 1 })
    .select("role content")
    .lean();

  const recentMessages = history.map((m) => ({
    role: m.role,
    content: m.content,
  }));

  // Save user message first
  await Message.create({
    sessionId,
    role: "user",
    content: message.trim(),
  });

  const pipelineBody = {
    static: {
      disease: session.staticContext.disease,
      intent: session.staticContext.intent,
      location: session.staticContext.location,
      patientName: session.staticContext.patientName,
    },
    dynamic: { recentMessages },
    current: { userMessage: message.trim() },
  };

  // Set SSE headers
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("X-Accel-Buffering", "no");
  res.flushHeaders();

  // Padding comment to bust edge-proxy buffering (Render/Cloudflare buffer
  // small chunks until ~2KB accumulates). A comment line is valid SSE that
  // clients ignore, but forces the proxy to flush subsequent chunks live.
  res.write(":" + " ".repeat(2048) + "\n\n");

  // Query-result cache check — skip whole pipeline on hit
  const ckey = cacheKey(
    session.staticContext.disease,
    session.staticContext.intent,
    message,
    recentMessages
  );
  const cachedResponse = await cacheGet(ckey);
  if (cachedResponse) {
    res.write(`event: status\ndata: {"stage":"cache_hit","message":"Served from cache"}\n\n`);
    res.write(`event: metadata\ndata: ${JSON.stringify(cachedResponse)}\n\n`);
    res.write(`event: done\ndata: {}\n\n`);

    await Message.create({
      sessionId,
      role: "assistant",
      content: cachedResponse.overview || JSON.stringify(cachedResponse),
      structuredResponse: cachedResponse,
      pipelineMeta: cachedResponse.pipelineMeta || null,
    });
    await Session.findByIdAndUpdate(sessionId, { $inc: { messageCount: 2 } });
    return res.end();
  }

  try {
    const resp = await fetch(`${FASTAPI_URL}/pipeline/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(pipelineBody),
    });

    if (!resp.ok) {
      res.write(`event: error\ndata: {"error":"pipeline returned ${resp.status}"}\n\n`);
      res.end();
      return;
    }

    let metadataJson = null;
    let sseBuffer = "";
    let currentEvent = null;

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      res.write(chunk);

      sseBuffer += chunk;
      const lines = sseBuffer.split("\n");
      sseBuffer = lines.pop() || "";

      for (const line of lines) {
        if (line.startsWith("event: ")) {
          currentEvent = line.slice(7).trim();
        } else if (line.startsWith("data: ") && currentEvent === "metadata") {
          try {
            metadataJson = JSON.parse(line.slice(6));
          } catch {
            // ignore — data may be across multiple lines in rare SSE flavors
          }
          currentEvent = null;
        } else if (line === "") {
          currentEvent = null;
        }
      }
    }

    // Save assistant message after stream completes
    if (metadataJson) {
      await Message.create({
        sessionId,
        role: "assistant",
        content: metadataJson.overview || JSON.stringify(metadataJson),
        structuredResponse: metadataJson,
        pipelineMeta: metadataJson.pipelineMeta || null,
      });

      await Session.findByIdAndUpdate(sessionId, {
        $inc: { messageCount: 2 },
      });

      if (!metadataJson.abstain_reason) {
        await cacheSet(ckey, metadataJson, CACHE_TTL_MS);
      }
    }
  } catch (err) {
    console.error(JSON.stringify({ id: req.id, level: "error", where: "pipeline_stream", error: err.message }));
    res.write(`event: error\ndata: {"error":"stream failed"}\n\n`);
  }

  res.end();
});

export default router;

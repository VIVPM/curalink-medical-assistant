// Authenticated chat routes for cached and streamed research responses.

import crypto from "crypto";
import { Router } from "express";
import rateLimit, { ipKeyGenerator } from "express-rate-limit";
import Session from "../models/Session.js";
import Message from "../models/Message.js";
import User from "../models/User.js";
import { cacheGet, cacheSet } from "../cache.js";
import { authMiddleware } from "../middleware/auth.js";

const router = Router();

const FASTAPI_URL = process.env.FASTAPI_URL || "http://localhost:8000";
const CACHE_TTL_MS = 24 * 60 * 60 * 1000;
const MAX_MESSAGE_LENGTH = 4000;
const DAILY_MESSAGE_CAP = Number(process.env.DAILY_MESSAGE_CAP) || 5;


const DAILY_TOKEN_CAP = Number(process.env.DAILY_TOKEN_CAP) || 0;




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

  if (_idempotencyStore.size > 10_000) {
    const now = Date.now();
    for (const [mk, mv] of _idempotencyStore) {
      if (now - mv.ts > IDEMPOTENCY_TTL_MS) _idempotencyStore.delete(mk);
    }
  }
}




// Counts today's user questions for daily quota enforcement.
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


// Estimates today's message tokens for optional token quotas.
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



const chatLimiter = rateLimit({
  windowMs: 60 * 1000,
  max: Number(process.env.CHAT_RATE_MAX) || 15,
  standardHeaders: true,
  legacyHeaders: false,
  keyGenerator: (req) => req.userId || ipKeyGenerator(req.ip),
  message: { ok: false, error: "rate limit exceeded, please slow down" },
});





// Normalizes cache-key text without collapsing distinct medical questions.
function normKey(s) {
  return (s || "")
    .toLowerCase()
    .replace(/['‘’]/g, "")
    .replace(/[^\w\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

export function cacheKey(userId, disease, intent, location, message, history = []) {
  const normalized = [userId, disease, intent, location, message]
    .map(normKey)
    .join("|");


  const historyStr = history
    .map((m) => `${m.role}:${normKey(m.content)}`)
    .join("|");
  const digest = crypto
    .createHash("sha256")
    .update(`${normalized}||${historyStr}`)
    .digest("hex");
  const tenant = String(userId).trim().toLowerCase();
  return `query:${tenant}:${digest}`;
}

export function isEmergencyMessage(message) {
  const text = message || "";
  return [
    /\b(overdose|suicidal|kill myself|self[- ]harm|anaphylaxis)\b/i,
    /\b(?:can't|cannot|unable to)\s+breathe\b/i,
    /\b(?:chest pain|severe bleeding|stroke symptoms|unconscious|seizure)\b.*\b(?:now|currently|right now)\b/i,
    /\b(?:now|currently|right now)\b.*\b(?:chest pain|severe bleeding|stroke symptoms|unconscious|seizure)\b/i,
  ].some((pattern) => pattern.test(text));
}

function emergencyResponse() {
  return {
    overview: "This may describe an urgent or emergency situation. Contact local emergency services now or go to the nearest emergency department.",
    insights: [],
    trials: [],
    recommendations: [],
    follow_up_questions: [],
    abstain_reason: "Curalink is not designed for emergencies or urgent medical assessment.",
    suggestion: "Do not wait for an online research response. If it is safe to do so, stay with the person until help arrives.",
    pipelineMeta: { safety: "emergency", warnings: [], citation_stats: { total: 0, verified: 0, unverified: 0 } },
  };
}


router.use(authMiddleware);
router.use(chatLimiter);


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



  const idempKey = req.headers["idempotency-key"];
  if (idempKey) {
    const prev = idempotencyCheck(req.userId, idempKey);
    if (prev) return res.json(prev);
  }


  const session = await Session.findOne({ _id: sessionId, userId: req.userId });
  if (!session) {
    return res.status(404).json({ ok: false, error: "session not found" });
  }

  if (isEmergencyMessage(message)) {
    const response = emergencyResponse();
    return res.json({
      ok: true,
      response,
      assistantMessage: { role: "assistant", content: response.overview, structuredResponse: response },
    });
  }



  const used = await messagesUsedToday(req.userId);
  if (used >= DAILY_MESSAGE_CAP) {

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


  const userMsg = await Message.create({
    sessionId,
    role: "user",
    content: message.trim(),
  });


  const ckey = cacheKey(
    req.userId.toString(),
    session.staticContext.disease,
    session.staticContext.intent,
    session.staticContext.location,
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


  const pipelineBody = {
    tenant: req.userId.toString(),
    static: {
      disease: session.staticContext.disease,
      intent: session.staticContext.intent,
      location: session.staticContext.location,
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
      headers: {
        "Content-Type": "application/json",
        "X-Request-Id": req.id,
        "X-Internal-API-Key": process.env.INTERNAL_API_KEY,
      },
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


  const assistantContent =
    pipelineResult.overview || JSON.stringify(pipelineResult);

  const assistantMsg = await Message.create({
    sessionId,
    role: "assistant",
    content: assistantContent,
    structuredResponse: pipelineResult,
    pipelineMeta: pipelineResult.pipelineMeta || null,
  });


  await Session.findByIdAndUpdate(sessionId, {
    $inc: { messageCount: 2 },
  });


  if (!pipelineResult.abstain_reason) {
    await cacheSet(ckey, pipelineResult, CACHE_TTL_MS);
  }


  const result = {
    ok: true,
    userMessage: userMsg,
    assistantMessage: assistantMsg,
    response: pipelineResult,
  };
  if (idempKey) idempotencySet(req.userId, idempKey, result);
  res.json(result);
});


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

  if (isEmergencyMessage(message)) {
    const response = emergencyResponse();
    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache, no-transform");
    res.setHeader("Connection", "keep-alive");
    res.flushHeaders();
    res.write(`event: metadata\ndata: ${JSON.stringify(response)}\n\n`);
    res.write("event: done\ndata: {}\n\n");
    return res.end();
  }


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


  await Message.create({
    sessionId,
    role: "user",
    content: message.trim(),
  });

  const pipelineBody = {
    tenant: req.userId.toString(),
    static: {
      disease: session.staticContext.disease,
      intent: session.staticContext.intent,
      location: session.staticContext.location,
    },
    dynamic: { recentMessages },
    current: { userMessage: message.trim() },
  };


  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("X-Accel-Buffering", "no");
  res.flushHeaders();




  res.write(":" + " ".repeat(2048) + "\n\n");


  const ckey = cacheKey(
    req.userId.toString(),
    session.staticContext.disease,
    session.staticContext.intent,
    session.staticContext.location,
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
      headers: {
        "Content-Type": "application/json",
        "X-Request-Id": req.id,
        "X-Internal-API-Key": process.env.INTERNAL_API_KEY,
      },
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
          } catch {}

          currentEvent = null;
        } else if (line === "") {
          currentEvent = null;
        }
      }
    }


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

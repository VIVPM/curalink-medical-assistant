import crypto from "crypto";
import { Router } from "express";
import Session from "../models/Session.js";
import Message from "../models/Message.js";
import Cache from "../models/Cache.js";
import { authMiddleware } from "../middleware/auth.js";

const router = Router();

const FASTAPI_URL = process.env.FASTAPI_URL || "http://localhost:8000";  // overridden by env at deploy
const CACHE_TTL_MS = 24 * 60 * 60 * 1000;

function cacheKey(disease, intent, message) {
  const normalized = `${(disease || "").trim().toLowerCase()}|${(intent || "").trim().toLowerCase()}|${message.trim().toLowerCase().replace(/\s+/g, " ")}`;
  return crypto.createHash("sha256").update(normalized).digest("hex");
}

// All chat routes require auth
router.use(authMiddleware);

// POST /api/chat — send message, run pipeline, return structured response
router.post("/chat", async (req, res) => {
  const { sessionId, message } = req.body;

  if (!sessionId) {
    return res.status(400).json({ ok: false, error: "sessionId is required" });
  }
  if (!message || !message.trim()) {
    return res.status(400).json({ ok: false, error: "message is required" });
  }

  // 1. Load session
  const session = await Session.findById(sessionId);
  if (!session) {
    return res.status(404).json({ ok: false, error: "session not found" });
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
    message
  );
  const cached = await Cache.findOne({ key: ckey }).lean();
  if (cached && cached.response) {
    const assistantMsg = await Message.create({
      sessionId,
      role: "assistant",
      content: cached.response.overview || JSON.stringify(cached.response),
      structuredResponse: cached.response,
      pipelineMeta: cached.response.pipelineMeta || null,
    });
    await Session.findByIdAndUpdate(sessionId, { $inc: { messageCount: 2 } });
    return res.json({
      ok: true,
      userMessage: userMsg,
      assistantMessage: assistantMsg,
      response: cached.response,
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
      const err = await resp.text();
      return res.status(502).json({
        ok: false,
        error: "pipeline failed",
        detail: err,
      });
    }

    pipelineResult = await resp.json();
  } catch (err) {
    return res.status(503).json({
      ok: false,
      error: "fastapi unreachable",
      detail: err.message,
    });
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
    await Cache.updateOne(
      { key: ckey },
      {
        key: ckey,
        response: pipelineResult,
        expiresAt: new Date(Date.now() + CACHE_TTL_MS),
      },
      { upsert: true }
    );
  }

  // 8. Return response
  res.json({
    ok: true,
    userMessage: userMsg,
    assistantMessage: assistantMsg,
    response: pipelineResult,
  });
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

  const session = await Session.findById(sessionId);
  if (!session) {
    return res.status(404).json({ ok: false, error: "session not found" });
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
    message
  );
  const cached = await Cache.findOne({ key: ckey }).lean();
  if (cached && cached.response) {
    res.write(`event: status\ndata: {"stage":"cache_hit","message":"Served from cache"}\n\n`);
    res.write(`event: metadata\ndata: ${JSON.stringify(cached.response)}\n\n`);
    res.write(`event: done\ndata: {}\n\n`);

    await Message.create({
      sessionId,
      role: "assistant",
      content: cached.response.overview || JSON.stringify(cached.response),
      structuredResponse: cached.response,
      pipelineMeta: cached.response.pipelineMeta || null,
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
        await Cache.updateOne(
          { key: ckey },
          {
            key: ckey,
            response: metadataJson,
            expiresAt: new Date(Date.now() + CACHE_TTL_MS),
          },
          { upsert: true }
        );
      }
    }
  } catch (err) {
    res.write(`event: error\ndata: {"error":"${err.message}"}\n\n`);
  }

  res.end();
});

export default router;

// Tests chat cache isolation and urgent-use routing.

import test from "node:test";
import assert from "node:assert/strict";
import chatRouter, { cacheKey, isEmergencyMessage } from "./chat.js";
import express from "express";
import { signToken } from "../middleware/auth.js";
import jobsRouter from "./jobs.js";

test("job polling does not consume chat quota and both chat routes remain limited", async () => {
  const oldSecret = process.env.JWT_SECRET;
  process.env.JWT_SECRET = "chat-limiter-regression-test";
  const app = express();
  app.use(express.json());
  app.use("/api", chatRouter);
  app.get("/api/jobs/:id", (req, res) => res.json({ state: "running" }));
  app.use("/api/jobs", jobsRouter);
  const server = app.listen(0, "127.0.0.1");
  await new Promise(resolve => server.once("listening", resolve));
  const url = `http://127.0.0.1:${server.address().port}`;
  const headers = { Authorization: `Bearer ${signToken("polling-regression")}`, "Content-Type": "application/json" };
  const limit = Number(process.env.CHAT_RATE_MAX) || 15;
  try {
    for (let i = 0; i < limit + 5; i++) {
      assert.equal((await fetch(`${url}/api/jobs/test`, { headers })).status, 200);
    }
    for (let i = 0; i < limit; i++) {
      const path = i % 2 ? "/chat/stream" : "/chat";
      assert.equal((await fetch(`${url}/api${path}`, { method: "POST", headers, body: "{}" })).status, 400);
    }
    const blocked = await fetch(`${url}/api/chat`, { method: "POST", headers, body: "{}" });
    assert.equal(blocked.status, 429);
    assert.ok(Number(blocked.headers.get("Retry-After")) > 0);
    assert.equal((await fetch(`${url}/api/chat/stream`, { method: "POST", headers, body: "{}" })).status, 429);
    assert.equal((await fetch(`${url}/api/jobs/test`, { headers })).status, 200);
    assert.equal((await fetch(`${url}/api/jobs`, { method: "POST", headers, body: "{}" })).status, 429);
    assert.equal((await fetch(`${url}/api/jobs/test`)).status, 401);
  } finally {
    await new Promise(resolve => server.close(resolve));
    if (oldSecret === undefined) delete process.env.JWT_SECRET;
    else process.env.JWT_SECRET = oldSecret;
  }
});

const history = [{ role: "user", content: "Earlier question" }];
const key = (user, location) =>
  cacheKey(user, "Parkinson's", "DBS", location, "Latest evidence?", history);

test("query cache is isolated by user and location", () => {
  assert.match(key("user-a", "Toronto"), /^query:user-a:/);
  assert.notEqual(key("user-a", "Toronto"), key("user-b", "Toronto"));
  assert.notEqual(key("user-a", "Toronto"), key("user-a", "Boston"));
  assert.equal(key("USER-A", " Toronto "), key("user-a", "toronto"));
});

test("query cache includes prior evidence", () => {
  const build = (snippet) => cacheKey("user-a", "Parkinson's", "DBS", "Toronto", "Explain that", [{
    role: "assistant", content: "Same overview", structuredResponse: {
      insights: [{ finding: "Same finding", source_details: [{ url: "https://example.org/study", snippet }] }],
    },
  }]);
  assert.notEqual(build("Evidence A"), build("Evidence B"));
  assert.match(build("Evidence A"), /^query:user-a:[a-f0-9]{64}$/);
});

test("urgent messages are diverted from the research pipeline", () => {
  assert.equal(isEmergencyMessage("I cannot breathe"), true);
  assert.equal(isEmergencyMessage("I took an overdose"), true);
  assert.equal(isEmergencyMessage("What does research say about asthma treatment?"), false);
});

// Tests signup consent requirements.

import test from "node:test";
import assert from "node:assert/strict";
import express from "express";
import authRouter from "./auth.js";

async function withServer(run) {
  const app = express();
  app.use(express.json());
  app.use(authRouter);
  const server = app.listen(0);
  await new Promise((resolve) => server.on("listening", resolve));
  try {
    await run(server.address().port);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
}

test("signup requires explicit terms acceptance", async () => {
  await withServer(async (port) => {
    const response = await fetch(`http://127.0.0.1:${port}/signup`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: "Research User", email: "research@example.com", password: "password123" }),
    });
    assert.equal(response.status, 400);
    assert.equal((await response.json()).error, "terms and privacy notice must be accepted");
  });
});

test("session checks are not throttled by the credential rate limit", async () => {
  await withServer(async (port) => {
    const limit = Number(process.env.AUTH_RATE_MAX) || 20;
    for (let i = 0; i < limit + 5; i++) {
      const response = await fetch(`http://127.0.0.1:${port}/me`);
      assert.equal(response.status, 401);
    }
  });
});

test("refresh rejects access tokens and malformed refresh tokens", async () => {
  process.env.JWT_SECRET = "auth-refresh-test-secret";
  const { signToken } = await import("../middleware/auth.js");
  await withServer(async (port) => {
    for (const refreshToken of [signToken("user-a"), "not-a-token"]) {
      const response = await fetch(`http://127.0.0.1:${port}/refresh`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ refreshToken }),
      });
      assert.equal(response.status, 401);
    }
  });
});

test("signup is still rate limited", async () => {
  await withServer(async (port) => {
    const limit = Number(process.env.AUTH_RATE_MAX) || 20;
    let last;
    for (let i = 0; i <= limit; i++) {
      last = await fetch(`http://127.0.0.1:${port}/signup`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
    }
    assert.equal(last.status, 429);
  });
});

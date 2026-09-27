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

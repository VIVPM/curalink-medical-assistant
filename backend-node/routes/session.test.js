// Tests session ownership enforcement.

import test from "node:test";
import assert from "node:assert/strict";
import express from "express";
import sessionRouter from "./session.js";
import { db } from "../db.js";
import { signToken } from "../middleware/auth.js";

process.env.JWT_SECRET = "session-route-test-secret";

test("session lookup is scoped to the authenticated user", async () => {
  const originalQuery = db.query;
  let call;
  db.query = async (text, params) => {
    call = { text, params };
    return { rows: [], rowCount: 0 };
  };

  const app = express();
  app.use(express.json());
  app.use(sessionRouter);
  const server = app.listen(0);
  await new Promise((resolve) => server.on("listening", resolve));

  try {
    const { port } = server.address();
    const response = await fetch(`http://127.0.0.1:${port}/session/another-users-session`, {
      headers: { Authorization: `Bearer ${signToken("user-a")}` },
    });
    assert.equal(response.status, 404);
    assert.match(call.text, /WHERE id = \$1 AND user_id = \$2/);
    assert.deepEqual(call.params, ["another-users-session", "user-a"]);
  } finally {
    db.query = originalQuery;
    await new Promise((resolve) => server.close(resolve));
  }
});

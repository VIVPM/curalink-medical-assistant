// Tests session ownership enforcement.

import test from "node:test";
import assert from "node:assert/strict";
import express from "express";
import sessionRouter from "./session.js";
import Session from "../models/Session.js";
import { signToken } from "../middleware/auth.js";

process.env.JWT_SECRET = "session-route-test-secret";

test("session lookup is scoped to the authenticated user", async () => {
  const originalFindOne = Session.findOne;
  let query;
  Session.findOne = (value) => {
    query = value;
    return { lean: async () => null };
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
    assert.equal(query.userId, "user-a");
    assert.equal(query._id, "another-users-session");
  } finally {
    Session.findOne = originalFindOne;
    await new Promise((resolve) => server.close(resolve));
  }
});

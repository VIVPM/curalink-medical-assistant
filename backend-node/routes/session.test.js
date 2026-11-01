// Tests session ownership enforcement.

import test from "node:test";
import assert from "node:assert/strict";
import express from "express";
import sessionRouter from "./session.js";
import { db, SCHEMA, toSession } from "../db.js";
import { signToken } from "../middleware/auth.js";

process.env.JWT_SECRET = "session-route-test-secret";

test("optional patient names persist as private session metadata, not research context", async () => {
  const originalQuery = db.query;
  const rows = new Map();
  db.query = async (text, params) => {
    if (text.includes("INSERT INTO sessions")) {
      const [id, user_id, disease, intent, location, title, patient_name] = params;
      const row = { id, user_id, disease, intent, location, title, patient_name };
      rows.set(id, row);
      return { rows: [row] };
    }
    if (text.startsWith("SELECT * FROM sessions")) {
      const row = rows.get(params[0]);
      return { rows: row?.user_id === params[1] ? [row] : [] };
    }
    if (text.startsWith("SELECT * FROM messages")) return { rows: [] };
    throw new Error("Unexpected test query");
  };
  const app = express();
  app.use(express.json());
  app.use(sessionRouter);
  const server = app.listen(0);
  await new Promise(resolve => server.once("listening", resolve));
  const headers = { Authorization: `Bearer ${signToken("user-a")}`, "Content-Type": "application/json" };
  const url = `http://127.0.0.1:${server.address().port}`;
  try {
    const create = await fetch(`${url}/session`, { method: "POST", headers,
      body: JSON.stringify({ disease: "Asthma", patientName: "  Patient A  ", userId: "user-b" }) });
    assert.equal(create.status, 201);
    const { session } = await create.json();
    assert.equal(session.patientName, "Patient A");
    assert.equal(session.userId, "user-a");
    assert.equal(rows.get(session._id).patient_name, "Patient A");
    assert.equal(JSON.stringify(session.staticContext).includes("Patient A"), false);
    assert.equal(session.title.includes("Patient A"), false);
    const loaded = await fetch(`${url}/session/${session._id}`, { headers });
    assert.equal((await loaded.json()).session.patientName, "Patient A");
    const other = await fetch(`${url}/session/${session._id}`, {
      headers: { Authorization: `Bearer ${signToken("user-b")}` },
    });
    assert.equal(other.status, 404);
    const blank = await fetch(`${url}/session`, { method: "POST", headers, body: JSON.stringify({ disease: "Asthma" }) });
    assert.equal(blank.status, 201);
    assert.equal((await blank.json()).session.patientName, "");
    for (const patientName of [null, {}, 12, "x".repeat(101)]) {
      const invalid = await fetch(`${url}/session`, { method: "POST", headers, body: JSON.stringify({ disease: "Asthma", patientName }) });
      assert.equal(invalid.status, 400);
    }
    assert.equal(toSession({ id: "legacy" }).patientName, "");
    assert.match(SCHEMA, /ALTER TABLE sessions ADD COLUMN IF NOT EXISTS patient_name/);
  } finally {
    db.query = originalQuery;
    await new Promise(resolve => server.close(resolve));
  }
});

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

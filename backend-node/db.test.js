// Tests data-retention purge windows and account consent columns.

import test from "node:test";
import assert from "node:assert/strict";
import { db, purgeExpired, SCHEMA } from "./db.js";

const DAY_MS = 24 * 60 * 60 * 1000;

test("purge removes sessions and messages after 90 days and audit logs after 1 year", async () => {
  const original = db.query;
  const calls = [];
  db.query = async (text, params) => calls.push({ text, params });
  try {
    await purgeExpired();
  } finally {
    db.query = original;
  }
  const ageDays = (table) => {
    const call = calls.find((c) => c.text.startsWith(`DELETE FROM ${table} `));
    return Math.round((Date.now() - call.params[0]) / DAY_MS);
  };
  assert.equal(ageDays("messages"), 90);
  assert.equal(ageDays("sessions"), 90);
  assert.equal(ageDays("audit_logs"), 365);
  assert.ok(calls.some((c) => c.text.startsWith("DELETE FROM caches ")));
});

test("account deletion cascades and terms acceptance is recorded", () => {
  for (const table of ["sessions", "messages", "audit_logs", "webhooks"]) {
    const ddl = SCHEMA.split("CREATE TABLE IF NOT EXISTS ").find((t) => t.startsWith(`${table} `));
    assert.match(ddl, /REFERENCES \w+\(id\) ON DELETE CASCADE/);
  }
  assert.match(SCHEMA, /terms_accepted_at TIMESTAMPTZ/);
  assert.match(SCHEMA, /terms_version TEXT/);
});

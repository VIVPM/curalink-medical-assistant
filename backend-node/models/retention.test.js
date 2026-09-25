import test from "node:test";
import assert from "node:assert/strict";
import Message from "./Message.js";
import Session from "./Session.js";
import User from "./User.js";

const ttlSeconds = 90 * 24 * 60 * 60;
const hasTtl = (model, field) =>
  model.schema.indexes().some(([keys, options]) => keys[field] === 1 && options.expireAfterSeconds === ttlSeconds);

test("sessions and messages have 90-day TTL indexes", () => {
  assert.equal(hasTtl(Session, "updatedAt"), true);
  assert.equal(hasTtl(Message, "createdAt"), true);
});

test("terms acceptance is recorded on user accounts", () => {
  assert.ok(User.schema.path("termsAcceptedAt"));
  assert.ok(User.schema.path("termsVersion"));
});

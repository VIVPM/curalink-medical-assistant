// Tests user-owned key parsing and forwarding.

import test from "node:test";
import assert from "node:assert/strict";
import { ownKeysFromRequest, providerHeaders, validateOwnKeys } from "./own_keys.js";

test("requests without provider keys keep default credits", () => {
  assert.equal(ownKeysFromRequest({ headers: {} }), null);
});

test("provider keys are parsed and forwarded without extra fields", () => {
  const keys = ownKeysFromRequest({ headers: { "x-provider-hf-key": "hf_user" } });
  assert.deepEqual(providerHeaders(keys), {
    "X-Provider-HF-Key": "hf_user",
    "X-Provider-CF-Account": "",
    "X-Provider-CF-Key": "",
  });
});

test("oversized provider keys are rejected", () => {
  assert.throws(
    () => ownKeysFromRequest({ headers: { "x-provider-hf-key": "x".repeat(1025) } }),
    { status: 400 },
  );
});

test("rejected provider validation returns a client error", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => new Response("{}", { status: 400 });
  try {
    await assert.rejects(validateOwnKeys({ hfToken: "bad" }), { status: 400 });
  } finally {
    globalThis.fetch = originalFetch;
  }
});

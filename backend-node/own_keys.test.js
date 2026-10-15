// Tests user-owned key parsing and forwarding.

import test from "node:test";
import assert from "node:assert/strict";
import { ownKeysFromRequest, providerHeaders, validateOwnKeys } from "./own_keys.js";

test("requests without provider keys keep default credits", () => {
  assert.equal(ownKeysFromRequest({ headers: {} }), null);
});

test("provider selection and keys are forwarded without extra fields", () => {
  const keys = ownKeysFromRequest({ headers: {
    "x-provider-name": "huggingface", "x-provider-hf-key": "hf_user",
  } });
  assert.deepEqual(providerHeaders(keys), {
    "X-Provider-Name": "huggingface",
    "X-Provider-HF-Key": "hf_user",
    "X-Provider-CF-Account": "",
    "X-Provider-CF-Key": "",
  });
});

test("incomplete provider keys cannot bypass credits", () => {
  for (const headers of [
    { "x-provider-hf-key": "hf_user" },
    { "x-provider-name": "cloudflare", "x-provider-hf-key": "hf_user" },
    { "x-provider-name": "unknown", "x-provider-hf-key": "hf_user" },
  ]) {
    assert.throws(() => ownKeysFromRequest({ headers }), { status: 400 });
  }
});

test("oversized provider keys are rejected", () => {
  assert.throws(
    () => ownKeysFromRequest({ headers: { "x-provider-name": "huggingface", "x-provider-hf-key": "x".repeat(1025) } }),
    { status: 400 },
  );
});

test("rejected provider validation returns a client error", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => new Response("{}", { status: 400 });
  try {
    await assert.rejects(validateOwnKeys({ provider: "huggingface", hfToken: "bad" }), { status: 400 });
  } finally {
    globalThis.fetch = originalFetch;
  }
});

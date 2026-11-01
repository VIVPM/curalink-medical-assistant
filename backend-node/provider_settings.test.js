// Saved-provider lifecycle, encryption, and tenant isolation.

import test from "node:test";
import assert from "node:assert/strict";
import express from "express";
import { db, SCHEMA } from "./db.js";
import { signToken } from "./middleware/auth.js";
import providerRouter from "./routes/provider.js";
import jobCredentialsRouter from "./routes/job_credentials.js";
import {
  decryptCredential, deleteProviderKeys, encryptCredential, isUnlimited,
  providerSettings, publicSettings, requestOwnKeys, saveProviderKeys, selectProvider,
} from "./provider_settings.js";

process.env.JWT_SECRET = "provider-storage-test-secret";
process.env.INTERNAL_API_KEY = "worker-lookup-test-secret";
process.env.FASTAPI_URL = "https://orchestrator.example";
const { default: jobsRouter } = await import("./routes/jobs.js");

function memoryDatabase() {
  const rows = new Map();
  const calls = [];
  const query = async (sql, params) => {
    calls.push({ sql, params });
    const [id, selected, hf, account, cf] = params;
    if (sql.startsWith("SELECT * FROM user_provider_settings")) {
      return { rows: rows.has(id) ? [{ ...rows.get(id) }] : [] };
    }
    const row = rows.get(id) || { user_id: id, selected: "free" };
    if (sql.includes("INSERT INTO user_provider_settings")) {
      row.selected = selected;
      if (params.length === 5) {
        row.hf_token_enc = hf;
        if (account !== null) row.cf_account_id = account;
        if (cf !== null) row.cf_token_enc = cf;
      }
    } else if (sql.includes("hf_token_enc = NULL")) {
      row.hf_token_enc = null;
      row.selected = "free";
    } else if (sql.includes("cf_account_id = NULL")) {
      row.cf_account_id = null;
      row.cf_token_enc = null;
      if (row.selected === "cloudflare") row.selected = "free";
    } else {
      throw new Error(`Unexpected test query: ${sql}`);
    }
    rows.set(id, row);
    return { rows: [], rowCount: 1 };
  };
  return { rows, calls, query };
}

async function withStore(run) {
  const originalQuery = db.query;
  const originalFetch = globalThis.fetch;
  const store = memoryDatabase();
  const validations = [];
  const queued = [];
  const forwarded = [];
  db.query = store.query;
  globalThis.fetch = async (url, options) => {
    if (String(url).endsWith("/keys/validate")) {
      validations.push(JSON.parse(options.body));
      return Response.json({ ok: true });
    }
    if (options?.headers?.["X-Tenant-ID"] && options.method === "POST") {
      queued.push({ body: JSON.parse(options.body), headers: options.headers });
      return Response.json({ ok: true, job_id: "queued-test-job" }, { status: 202 });
    }
    if (String(url).startsWith(`${process.env.FASTAPI_URL}/jobs/`) && options?.headers?.["X-Tenant-ID"]) {
      forwarded.push(options.headers);
      return Response.json({ detail: "Job not found" }, { status: 404 });
    }
    return originalFetch(url, options);
  };
  try { await run({ ...store, validations, queued, forwarded }); }
  finally { db.query = originalQuery; globalThis.fetch = originalFetch; }
}

test("credential encryption authenticates user, field and ciphertext", () => {
  const encrypted = encryptCredential("hf-private-value", "alice", "hf");
  assert.notEqual(encrypted, encryptCredential("hf-private-value", "alice", "hf"));
  assert.equal(encrypted.includes("hf-private-value"), false);
  assert.equal(decryptCredential(encrypted, "alice", "hf"), "hf-private-value");
  assert.equal(decryptCredential(encrypted, "bob", "hf"), "");
  assert.equal(decryptCredential(encrypted, "alice", "cf"), "");
  const changed = Buffer.from(encrypted, "base64");
  changed[changed.length - 1] ^= 1;
  assert.equal(decryptCredential(changed.toString("base64"), "alice", "hf"), "");
  assert.equal(decryptCredential("invalid", "alice", "hf"), "");
  assert.match(SCHEMA, /user_provider_settings[\s\S]*user_id TEXT PRIMARY KEY REFERENCES users\(id\) ON DELETE CASCADE/);
});

test("save, switch, reuse and remove providers without browser secrets", async () => {
  await withStore(async ({ rows, validations }) => {
    assert.equal(isUnlimited(await providerSettings("alice")), false);
    await saveProviderKeys("alice", { provider: "huggingface", hfToken: "hf-private-alice" });
    assert.equal(isUnlimited(await providerSettings("alice")), true);
    assert.equal(rows.get("alice").hf_token_enc.includes("hf-private-alice"), false);
    assert.deepEqual(publicSettings(await providerSettings("alice")).huggingface, { saved: true, last4: "lice" });
    await selectProvider("alice", "free");
    assert.equal(isUnlimited(await providerSettings("alice")), false);
    assert.equal((await providerSettings("alice")).hfToken, "hf-private-alice");
    await saveProviderKeys("alice", { provider: "cloudflare", cfAccountId: "a".repeat(32), cfToken: "cf-private-alice" });
    assert.equal(validations.at(-1).hfToken, "hf-private-alice");
    assert.equal((await requestOwnKeys({ userId: "alice", headers: {} })).provider, "cloudflare");
    await selectProvider("alice", "huggingface");
    assert.equal((await providerSettings("alice")).cfToken, "cf-private-alice");
    await saveProviderKeys("alice", { provider: "huggingface", hfToken: "hf-replaced" });
    await selectProvider("alice", "cloudflare");
    assert.equal((await requestOwnKeys({ userId: "alice", headers: {} })).hfToken, "hf-replaced");
    assert.equal((await providerSettings("bob")).selected, "free");
    await deleteProviderKeys("alice", "huggingface");
    assert.equal((await providerSettings("alice")).selected, "free");
    assert.equal(publicSettings(await providerSettings("alice")).cloudflare.saved, true);
    await assert.rejects(selectProvider("alice", "cloudflare"), { status: 400 });
    await deleteProviderKeys("alice", "cloudflare");
    assert.equal(publicSettings(await providerSettings("alice")).cloudflare.saved, false);
  });
});

test("missing or unreadable selected credentials never become app-funded requests", async () => {
  await withStore(async ({ rows }) => {
    rows.set("alice", { selected: "huggingface", hf_token_enc: "unreadable" });
    await assert.rejects(requestOwnKeys({ userId: "alice", headers: {} }), { status: 400 });
    assert.equal(isUnlimited(await providerSettings("alice")), false);
    await selectProvider("alice", "free");
    assert.equal(await requestOwnKeys({ userId: "alice", headers: {} }), null);
    for (const body of [null, {}, { provider: "other" }, { provider: "huggingface", hfToken: {} },
      { provider: "huggingface", hfToken: "" }, { provider: "cloudflare", hfToken: 0 }]) {
      await assert.rejects(saveProviderKeys("bob", body), { status: 400 });
    }
    await assert.rejects(selectProvider("alice", "other"), { status: 400 });
    await assert.rejects(deleteProviderKeys("alice", "other"), { status: 400 });
  });
});

test("rejected validation cannot overwrite saved credentials or change billing", async () => {
  await withStore(async ({ rows }) => {
    await saveProviderKeys("alice", { provider: "huggingface", hfToken: "valid-original" });
    const before = { ...rows.get("alice") };
    globalThis.fetch = async () => Response.json({ ok: false }, { status: 400 });
    await assert.rejects(saveProviderKeys("alice", { provider: "huggingface", hfToken: "rejected" }), { status: 400 });
    assert.deepEqual(rows.get("alice"), before);
    await selectProvider("alice", "free");
    await assert.rejects(selectProvider("alice", "huggingface"), { status: 400 });
    assert.equal(rows.get("alice").selected, "free");
  });
});

test("account endpoints isolate users and queue only authenticated credential references", async () => {
  await withStore(async ({ queued, forwarded, rows }) => {
    const app = express();
    app.use(express.json());
    app.use("/account", providerRouter);
    app.use("/jobs", jobsRouter);
    app.use("/internal", jobCredentialsRouter);
    const server = app.listen(0);
    await new Promise(resolve => server.once("listening", resolve));
    const url = `http://127.0.0.1:${server.address().port}`;
    const headers = id => ({ Authorization: `Bearer ${signToken(id)}`, "Content-Type": "application/json" });
    try {
      assert.equal((await fetch(`${url}/account/api-keys`)).status, 401);
      const saved = await fetch(`${url}/account/api-keys`, { method: "PUT", headers: headers("alice"),
        body: JSON.stringify({ provider: "huggingface", hfToken: "hf-alice-secret", userId: "bob" }) });
      assert.equal(saved.status, 200);
      const text = await saved.text();
      assert.equal(text.includes("hf-alice-secret"), false);
      assert.equal(text.includes("hf_token_enc"), false);
      const other = await fetch(`${url}/account/api-keys`, { headers: headers("bob") });
      assert.equal((await other.json()).huggingface.saved, false);
      const job = await fetch(`${url}/jobs`, { method: "POST", headers: headers("alice"),
        body: JSON.stringify({ tenant: "bob", credential_provider: "free" }) });
      assert.equal(job.status, 202);
      assert.equal(queued[0].body.tenant, "alice");
      assert.equal(queued[0].body.credential_provider, "huggingface");
      assert.equal(JSON.stringify(queued).includes("hf-alice-secret"), false);
      const lookupBody = JSON.stringify({ tenant: "alice", provider: "huggingface" });
      const lookupHeaders = { "Content-Type": "application/json", "X-Internal-API-Key": process.env.INTERNAL_API_KEY };
      assert.equal((await fetch(`${url}/internal/job-credentials`, { method: "POST", headers: headers("alice"), body: lookupBody })).status, 401);
      const lookup = await fetch(`${url}/internal/job-credentials`, { method: "POST", headers: lookupHeaders, body: lookupBody });
      assert.equal(lookup.status, 200);
      assert.equal(lookup.headers.get("cache-control"), "no-store");
      assert.equal((await lookup.json()).hfToken, "hf-alice-secret");
      const mismatch = await fetch(`${url}/internal/job-credentials`, { method: "POST", headers: lookupHeaders,
        body: JSON.stringify({ tenant: "alice", provider: "cloudflare" }) });
      assert.equal(mismatch.status, 409);
      const originalCiphertext = rows.get("alice").hf_token_enc;
      rows.get("alice").hf_token_enc = "unreadable";
      assert.equal((await fetch(`${url}/internal/job-credentials`, { method: "POST", headers: lookupHeaders, body: lookupBody })).status, 400);
      rows.get("alice").hf_token_enc = originalCiphertext;
      await saveProviderKeys("alice", { provider: "cloudflare", cfAccountId: "a".repeat(32), cfToken: "cf-alice-secret" });
      const cfLookup = await fetch(`${url}/internal/job-credentials`, { method: "POST", headers: lookupHeaders,
        body: JSON.stringify({ tenant: "alice", provider: "cloudflare" }) });
      assert.equal((await cfLookup.json()).cfToken, "cf-alice-secret");
      assert.equal((await fetch(`${url}/internal/job-credentials`, { method: "POST", headers: lookupHeaders, body: lookupBody })).status, 409);
      const ephemeral = await fetch(`${url}/jobs`, { method: "POST",
        headers: { ...headers("alice"), "X-Provider-Name": "huggingface", "X-Provider-Hf-Key": "hf-ephemeral" }, body: "{}" });
      assert.equal(ephemeral.status, 400);
      const freeJob = await fetch(`${url}/jobs`, { method: "POST", headers: headers("bob"),
        body: JSON.stringify({ tenant: "alice", credential_provider: "cloudflare" }) });
      assert.equal(freeJob.status, 202);
      assert.equal(queued.at(-1).body.tenant, "bob");
      assert.equal(queued.at(-1).body.credential_provider, "free");
      for (const [method, path] of [["GET", "/jobs/foreign"], ["DELETE", "/jobs/foreign"], ["GET", "/jobs/foreign/events"]]) {
        const response = await fetch(`${url}${path}`, { method, headers: { ...headers("bob"), "X-Tenant-ID": "alice" } });
        assert.equal(response.status, 404);
        assert.equal(forwarded.at(-1)["X-Tenant-ID"], "bob");
      }
      const removed = await fetch(`${url}/account/api-keys/huggingface`, { method: "DELETE", headers: headers("alice") });
      assert.equal(removed.status, 200);
      assert.equal((await removed.json()).selected, "free");
      assert.equal((await fetch(`${url}/internal/job-credentials`, { method: "POST", headers: lookupHeaders, body: lookupBody })).status, 409);
    } finally { await new Promise(resolve => server.close(resolve)); }
  });
});

// Account-backed provider credentials; only masked settings leave the API.

import crypto from "crypto";
import { db } from "./db.js";
import { ownKeysFromRequest, validateOwnKeys } from "./own_keys.js";

function encryptionKey() {
  if (!process.env.JWT_SECRET) throw new Error("JWT_SECRET is required for credential encryption");
  return crypto.createHash("sha256").update(`curalink-provider-keys\0${process.env.JWT_SECRET}`).digest();
}

export function encryptCredential(value, userId, field) {
  const iv = crypto.randomBytes(12);
  const cipher = crypto.createCipheriv("aes-256-gcm", encryptionKey(), iv);
  cipher.setAAD(Buffer.from(`${userId}:${field}`));
  const ciphertext = Buffer.concat([cipher.update(value, "utf8"), cipher.final()]);
  return Buffer.concat([iv, cipher.getAuthTag(), ciphertext]).toString("base64");
}

export function decryptCredential(value, userId, field) {
  if (!value) return "";
  try {
    const bytes = Buffer.from(value, "base64");
    const decipher = crypto.createDecipheriv("aes-256-gcm", encryptionKey(), bytes.subarray(0, 12));
    decipher.setAAD(Buffer.from(`${userId}:${field}`));
    decipher.setAuthTag(bytes.subarray(12, 28));
    return Buffer.concat([decipher.update(bytes.subarray(28)), decipher.final()]).toString("utf8");
  } catch {
    return "";
  }
}

export async function providerSettings(userId) {
  const { rows } = await db.query("SELECT * FROM user_provider_settings WHERE user_id = $1", [userId]);
  const row = rows[0];
  return {
    selected: row?.selected || "free",
    hfToken: decryptCredential(row?.hf_token_enc, userId, "hf"),
    cfAccountId: row?.cf_account_id || "",
    cfToken: decryptCredential(row?.cf_token_enc, userId, "cf"),
  };
}

export function publicSettings(settings) {
  return {
    selected: settings.selected,
    huggingface: { saved: Boolean(settings.hfToken), last4: settings.hfToken.slice(-4) || null },
    cloudflare: {
      saved: Boolean(settings.cfAccountId && settings.cfToken),
      last4: settings.cfToken.slice(-4) || null,
      accountId: settings.cfAccountId,
    },
  };
}

export function isUnlimited(settings) {
  return settings.selected === "huggingface" ? Boolean(settings.hfToken)
    : settings.selected === "cloudflare" && Boolean(settings.hfToken && settings.cfAccountId && settings.cfToken);
}

export function selectedKeys(settings) {
  if (settings.selected === "free") return null;
  if (!settings.hfToken || (settings.selected === "cloudflare" && (!settings.cfAccountId || !settings.cfToken))) {
    throw Object.assign(new Error("Saved API credentials are unavailable. Update them in Settings or select free usage."), { status: 400 });
  }
  return {
    provider: settings.selected,
    hfToken: settings.hfToken,
    cfAccountId: settings.selected === "cloudflare" ? settings.cfAccountId : "",
    cfToken: settings.selected === "cloudflare" ? settings.cfToken : "",
  };
}

export async function requestOwnKeys(req) {
  // Keep existing API clients working; the browser no longer stores or forwards secrets.
  return ownKeysFromRequest(req) || selectedKeys(await providerSettings(req.userId));
}

export async function saveProviderKeys(userId, body) {
  if (!body || !["huggingface", "cloudflare"].includes(body.provider)) {
    throw Object.assign(new Error("Unknown provider"), { status: 400 });
  }
  if (["hfToken", "cfAccountId", "cfToken"].some((field) => body[field] !== undefined
    && (typeof body[field] !== "string" || body[field].length > 1024))) {
    throw Object.assign(new Error("Invalid API credentials"), { status: 400 });
  }
  const settings = await providerSettings(userId);
  const keys = ownKeysFromRequest({ headers: {
    "x-provider-name": body.provider,
    "x-provider-hf-key": body.hfToken || (body.provider === "cloudflare" ? settings.hfToken : ""),
    "x-provider-cf-account": body.provider === "cloudflare" ? body.cfAccountId : "",
    "x-provider-cf-key": body.provider === "cloudflare" ? body.cfToken : "",
  } });
  await validateOwnKeys(keys);
  await db.query(
    `INSERT INTO user_provider_settings (user_id, selected, hf_token_enc, cf_account_id, cf_token_enc)
     VALUES ($1, $2, $3, $4, $5)
     ON CONFLICT (user_id) DO UPDATE SET selected = EXCLUDED.selected, hf_token_enc = EXCLUDED.hf_token_enc,
       cf_account_id = COALESCE(EXCLUDED.cf_account_id, user_provider_settings.cf_account_id),
       cf_token_enc = COALESCE(EXCLUDED.cf_token_enc, user_provider_settings.cf_token_enc)`,
    [userId, keys.provider, encryptCredential(keys.hfToken, userId, "hf"),
      keys.provider === "cloudflare" ? keys.cfAccountId : null,
      keys.provider === "cloudflare" ? encryptCredential(keys.cfToken, userId, "cf") : null]
  );
}

export async function selectProvider(userId, provider) {
  if (!["free", "huggingface", "cloudflare"].includes(provider)) {
    throw Object.assign(new Error("Unknown provider"), { status: 400 });
  }
  const settings = await providerSettings(userId);
  const keys = selectedKeys({ ...settings, selected: provider });
  if (keys) await validateOwnKeys(keys);
  await db.query(
    `INSERT INTO user_provider_settings (user_id, selected) VALUES ($1, $2)
     ON CONFLICT (user_id) DO UPDATE SET selected = EXCLUDED.selected`,
    [userId, provider]
  );
}

export async function deleteProviderKeys(userId, provider) {
  if (!["huggingface", "cloudflare"].includes(provider)) {
    throw Object.assign(new Error("Unknown provider"), { status: 400 });
  }
  if (provider === "huggingface") {
    await db.query("UPDATE user_provider_settings SET hf_token_enc = NULL, selected = 'free' WHERE user_id = $1", [userId]);
  } else {
    await db.query(
      `UPDATE user_provider_settings SET cf_account_id = NULL, cf_token_enc = NULL,
       selected = CASE WHEN selected = 'cloudflare' THEN 'free' ELSE selected END WHERE user_id = $1`, [userId]
    );
  }
}

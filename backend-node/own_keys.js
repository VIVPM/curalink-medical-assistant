// Validates ephemeral user inference keys with FastAPI without persisting them.

const FASTAPI_URL = process.env.FASTAPI_URL || "http://localhost:8000";

export function ownKeysFromRequest(req) {
  const hfToken = req.headers["x-provider-hf-key"] || "";
  const cfAccountId = req.headers["x-provider-cf-account"] || "";
  const cfToken = req.headers["x-provider-cf-key"] || "";
  if (![hfToken, cfAccountId, cfToken].some(Boolean)) return null;
  if ([hfToken, cfAccountId, cfToken].some((value) => typeof value !== "string" || value.length > 1024)) {
    throw Object.assign(new Error("Invalid API credentials"), { status: 400 });
  }
  return { hfToken, cfAccountId, cfToken };
}

export async function validateOwnKeys(keys) {
  let response;
  try {
    response = await fetch(`${FASTAPI_URL}/keys/validate`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Internal-API-Key": process.env.INTERNAL_API_KEY,
      },
      body: JSON.stringify(keys),
      signal: AbortSignal.timeout(15_000),
    });
  } catch {
    throw Object.assign(new Error("Credential verification unavailable"), { status: 503 });
  }
  if (!response.ok) {
    const status = response.status === 400 || response.status === 422 ? 400 : 503;
    throw Object.assign(new Error(status === 400 ? "Invalid or incomplete API credentials" : "Credential verification unavailable"), { status });
  }
}

export function providerHeaders(keys) {
  return keys ? {
    "X-Provider-HF-Key": keys.hfToken,
    "X-Provider-CF-Account": keys.cfAccountId,
    "X-Provider-CF-Key": keys.cfToken,
  } : {};
}

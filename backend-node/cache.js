// Query-result cache. Uses Redis when REDIS_URL is set, else falls back to the
// MongoDB Cache model — so the app runs identically with or without Redis.
import Redis from "ioredis";
import Cache from "./models/Cache.js";

const REDIS_URL = process.env.REDIS_URL;

let redis = null;
if (REDIS_URL) {
  redis = new Redis(REDIS_URL, {
    maxRetriesPerRequest: 2,
    enableOfflineQueue: false,
  });
  redis.on("connect", () => console.log("[cache] Redis connected (query cache on)"));
  redis.on("error", (e) =>
    console.error(JSON.stringify({ level: "error", where: "redis", error: e.message }))
  );
} else {
  console.log("[cache] REDIS_URL not set — query cache using MongoDB");
}

// Returns the cached response object, or null on miss / error.
export async function cacheGet(key) {
  if (redis) {
    try {
      const v = await redis.get(key);
      return v ? JSON.parse(v) : null;
    } catch (e) {
      console.error(JSON.stringify({ level: "error", where: "redis_get", error: e.message }));
      return null; // fail open — a cache miss just re-runs the pipeline
    }
  }
  const doc = await Cache.findOne({ key }).lean();
  return doc && doc.response ? doc.response : null;
}

// "disabled" (no REDIS_URL) | ioredis status ("ready" once connected) — for /health.
export function redisStatus() {
  if (!redis) return "disabled";
  return redis.status;
}

export async function cacheSet(key, value, ttlMs) {
  if (redis) {
    try {
      await redis.set(key, JSON.stringify(value), "PX", ttlMs);
    } catch (e) {
      console.error(JSON.stringify({ level: "error", where: "redis_set", error: e.message }));
    }
    return;
  }
  await Cache.updateOne(
    { key },
    { key, response: value, expiresAt: new Date(Date.now() + ttlMs) },
    { upsert: true }
  );
}

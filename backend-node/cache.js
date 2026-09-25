// Query-response cache backed by Redis with a MongoDB fallback.

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
      return null;
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

export async function cacheDeleteUser(userId) {
  const tenant = String(userId).toLowerCase();
  const patterns = [`query:${tenant}:*`, `semq:${tenant}:*`];

  if (redis) {
    for (const pattern of patterns) {
      let cursor = "0";
      do {
        const [next, keys] = await redis.scan(cursor, "MATCH", pattern, "COUNT", 100);
        cursor = next;
        if (keys.length) await redis.del(...keys);
      } while (cursor !== "0");
    }
    return;
  }

  await Cache.deleteMany({ key: { $regex: `^query:${tenant}:` } });
}

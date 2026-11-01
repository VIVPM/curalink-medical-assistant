// Worker-only credential lookup; secrets never enter queued job payloads.

import crypto from "crypto";
import { Router } from "express";
import { providerSettings, selectedKeys } from "../provider_settings.js";
import { validateOwnKeys } from "../own_keys.js";

const router = Router();

router.post("/job-credentials", async (req, res) => {
  res.set("Cache-Control", "no-store");
  const supplied = req.headers["x-internal-api-key"];
  const expected = process.env.INTERNAL_API_KEY;
  if (!expected || typeof supplied !== "string" || Buffer.byteLength(supplied) !== Buffer.byteLength(expected)
    || !crypto.timingSafeEqual(Buffer.from(supplied), Buffer.from(expected))) {
    return res.status(401).json({ ok: false, error: "Not authorized" });
  }
  const { tenant, provider } = req.body || {};
  if (typeof tenant !== "string" || !tenant || tenant.length > 128 || !["huggingface", "cloudflare"].includes(provider)) {
    return res.status(400).json({ ok: false, error: "Invalid credential reference" });
  }
  try {
    const settings = await providerSettings(tenant);
    if (settings.selected !== provider) {
      return res.status(409).json({ ok: false, error: "Saved provider changed or account removed. Submit a new job." });
    }
    const keys = selectedKeys(settings);
    await validateOwnKeys(keys);
    res.json(keys);
  } catch (error) {
    res.status(error.status || 503).json({ ok: false, error: "Saved provider credentials unavailable. Update Settings and submit a new job." });
  }
});

export default router;

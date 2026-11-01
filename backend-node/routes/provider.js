// Saved provider selection and encrypted account credentials.

import { Router } from "express";
import rateLimit, { ipKeyGenerator } from "express-rate-limit";
import { authMiddleware } from "../middleware/auth.js";
import { deleteProviderKeys, providerSettings, publicSettings, saveProviderKeys, selectProvider } from "../provider_settings.js";

const router = Router();
const keyLimiter = rateLimit({
  windowMs: 60 * 1000,
  max: 10,
  standardHeaders: true,
  legacyHeaders: false,
  keyGenerator: (req) => req.userId || ipKeyGenerator(req.ip),
  message: { ok: false, error: "Too many credential changes. Try again later." },
});

router.get("/api-keys", authMiddleware, async (req, res) => {
  res.json({ ok: true, ...publicSettings(await providerSettings(req.userId)) });
});

router.put("/api-keys", authMiddleware, keyLimiter, async (req, res) => {
  try {
    await saveProviderKeys(req.userId, req.body);
    res.json({ ok: true, ...publicSettings(await providerSettings(req.userId)) });
  } catch (error) {
    if (!error.status) throw error;
    res.status(error.status).json({ ok: false, error: error.message });
  }
});

router.put("/provider", authMiddleware, keyLimiter, async (req, res) => {
  try {
    await selectProvider(req.userId, req.body?.provider);
    res.json({ ok: true, ...publicSettings(await providerSettings(req.userId)) });
  } catch (error) {
    if (!error.status) throw error;
    res.status(error.status).json({ ok: false, error: error.message });
  }
});

router.delete("/api-keys/:provider", authMiddleware, keyLimiter, async (req, res) => {
  try {
    await deleteProviderKeys(req.userId, req.params.provider);
    res.json({ ok: true, ...publicSettings(await providerSettings(req.userId)) });
  } catch (error) {
    if (!error.status) throw error;
    res.status(error.status).json({ ok: false, error: error.message });
  }
});

export default router;

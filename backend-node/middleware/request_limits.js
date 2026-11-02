// Limit answer submissions, not status polling or unrelated API routes.
import rateLimit, { ipKeyGenerator } from "express-rate-limit";

export const answerSubmissionLimiter = rateLimit({
  windowMs: 60 * 1000,
  max: Number(process.env.CHAT_RATE_MAX) || 15,
  standardHeaders: true,
  legacyHeaders: false,
  keyGenerator: (req) => req.userId || ipKeyGenerator(req.ip),
  message: { ok: false, error: "rate limit exceeded, please slow down" },
});

import jwt from "jsonwebtoken";

// Read at call time (after dotenv has loaded) with NO fallback. A missing secret
// is a hard boot failure in index.js (SEC-2), so this is always set at runtime.
function jwtSecret() {
  const s = process.env.JWT_SECRET;
  if (!s) throw new Error("JWT_SECRET is not set");
  return s;
}

export function signToken(userId) {
  return jwt.sign({ userId }, jwtSecret(), { expiresIn: "1h" });
}

export function authMiddleware(req, res, next) {
  const header = req.headers.authorization;
  if (!header || !header.startsWith("Bearer ")) {
    return res.status(401).json({ ok: false, error: "not authenticated" });
  }

  try {
    const token = header.slice(7);
    const decoded = jwt.verify(token, jwtSecret());
    req.userId = decoded.userId;
    next();
  } catch {
    return res.status(401).json({ ok: false, error: "invalid token" });
  }
}

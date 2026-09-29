// JWT creation and authentication middleware.

import jwt from "jsonwebtoken";


// Returns the required JWT signing secret.
function jwtSecret() {
  const s = process.env.JWT_SECRET;
  if (!s) throw new Error("JWT_SECRET is not set");
  return s;
}

export function signToken(userId, authVersion = 0) {
  return jwt.sign({ userId, authVersion, type: "access" }, jwtSecret(), { expiresIn: "1h" });
}

export function signRefreshToken(userId, authVersion = 0) {
  return jwt.sign({ userId, authVersion, type: "refresh" }, jwtSecret(), { expiresIn: "7d" });
}

export function verifyRefreshToken(token) {
  const decoded = jwt.verify(token, jwtSecret());
  if (decoded.type !== "refresh") throw new Error("invalid refresh token");
  return decoded;
}

export function authMiddleware(req, res, next) {
  const header = req.headers.authorization;
  if (!header || !header.startsWith("Bearer ")) {
    return res.status(401).json({ ok: false, error: "not authenticated" });
  }

  try {
    const token = header.slice(7);
    const decoded = jwt.verify(token, jwtSecret());
    if (decoded.type !== "access") throw new Error("invalid access token");
    req.userId = decoded.userId;
    next();
  } catch {
    return res.status(401).json({ ok: false, error: "invalid token" });
  }
}

// Postgres (Neon) connection pool, schema bootstrap, retention purge, and row-to-API mappers.

import crypto from "crypto";
import pg from "pg";

const DAY_MS = 24 * 60 * 60 * 1000;
export const SESSION_RETENTION_DAYS = 90;
export const AUDIT_RETENTION_DAYS = 365;

let pool = null;

export const db = {
  query(text, params) {
    pool ??= new pg.Pool({ connectionString: process.env.DATABASE_URL, max: 10 });
    return pool.query(text, params);
  },
};

export const newId = () => crypto.randomBytes(12).toString("hex");

export const SCHEMA = `
CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  email TEXT NOT NULL UNIQUE,
  password TEXT NOT NULL,
  terms_accepted_at TIMESTAMPTZ,
  terms_version TEXT,
  auth_version INTEGER NOT NULL DEFAULT 0,
  credits INTEGER NOT NULL DEFAULT 5,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  disease TEXT NOT NULL,
  intent TEXT NOT NULL DEFAULT '',
  location TEXT NOT NULL DEFAULT '',
  title TEXT NOT NULL DEFAULT '',
  message_count INTEGER NOT NULL DEFAULT 0,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS sessions_user_id_idx ON sessions(user_id);
CREATE INDEX IF NOT EXISTS sessions_updated_at_idx ON sessions(updated_at);
CREATE TABLE IF NOT EXISTS messages (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
  content TEXT NOT NULL,
  own_key BOOLEAN NOT NULL DEFAULT false,
  structured_response JSONB,
  pipeline_meta JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS messages_session_created_idx ON messages(session_id, created_at);
CREATE INDEX IF NOT EXISTS messages_created_at_idx ON messages(created_at);
CREATE TABLE IF NOT EXISTS audit_logs (
  id TEXT PRIMARY KEY,
  user_id TEXT REFERENCES users(id) ON DELETE CASCADE,
  action TEXT NOT NULL,
  resource TEXT,
  resource_id TEXT,
  metadata JSONB,
  ip TEXT,
  timestamp TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS audit_logs_user_id_idx ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS audit_logs_action_idx ON audit_logs(action);
CREATE INDEX IF NOT EXISTS audit_logs_timestamp_idx ON audit_logs(timestamp);
CREATE TABLE IF NOT EXISTS caches (
  id TEXT PRIMARY KEY,
  key TEXT NOT NULL UNIQUE,
  response JSONB NOT NULL,
  expires_at TIMESTAMPTZ NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS caches_expires_at_idx ON caches(expires_at);
CREATE TABLE IF NOT EXISTS webhooks (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  url TEXT NOT NULL,
  secret TEXT NOT NULL,
  events TEXT[] NOT NULL DEFAULT ARRAY['job.completed'],
  active BOOLEAN NOT NULL DEFAULT true,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS webhooks_user_id_idx ON webhooks(user_id);
`;

/** Deletes rows past their retention window. */
export async function purgeExpired() {
  const sessionCutoff = new Date(Date.now() - SESSION_RETENTION_DAYS * DAY_MS);
  const auditCutoff = new Date(Date.now() - AUDIT_RETENTION_DAYS * DAY_MS);
  await db.query("DELETE FROM messages WHERE created_at < $1", [sessionCutoff]);
  await db.query("DELETE FROM sessions WHERE updated_at < $1", [sessionCutoff]);
  await db.query("DELETE FROM audit_logs WHERE timestamp < $1", [auditCutoff]);
  await db.query("DELETE FROM caches WHERE expires_at < now()");
}

/** Creates tables if missing and schedules an hourly retention purge. */
export async function initDb() {
  await db.query(SCHEMA);
  await purgeExpired();
  setInterval(() => purgeExpired().catch((e) => console.error(JSON.stringify({ level: "error", where: "purge_expired", error: e.message }))), 60 * 60 * 1000).unref();
}

export const closeDb = () => (pool ? pool.end() : Promise.resolve());

export const toUser = (r) =>
  r && {
    _id: r.id,
    name: r.name,
    email: r.email,
    termsAcceptedAt: r.terms_accepted_at,
    termsVersion: r.terms_version,
    authVersion: r.auth_version,
    credits: r.credits,
    createdAt: r.created_at,
    updatedAt: r.updated_at,
  };

export const toSession = (r) =>
  r && {
    _id: r.id,
    userId: r.user_id,
    staticContext: { disease: r.disease, intent: r.intent, location: r.location },
    title: r.title,
    messageCount: r.message_count,
    createdAt: r.created_at,
    updatedAt: r.updated_at,
  };

export const toMessage = (r) =>
  r && {
    _id: r.id,
    sessionId: r.session_id,
    role: r.role,
    content: r.content,
    ownKey: r.own_key,
    structuredResponse: r.structured_response,
    pipelineMeta: r.pipeline_meta,
    createdAt: r.created_at,
    updatedAt: r.updated_at,
  };

export const toWebhook = (r) =>
  r && {
    _id: r.id,
    userId: r.user_id,
    url: r.url,
    events: r.events,
    active: r.active,
    createdAt: r.created_at,
  };

import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import mongoose from "mongoose";
import authRouter from "./routes/auth.js";
import sessionRouter from "./routes/session.js";
import chatRouter from "./routes/chat.js";

dotenv.config();

const app = express();
const PORT = process.env.PORT || 4000;
const FASTAPI_URL = process.env.FASTAPI_URL || "http://localhost:8000";
const MONGO_URI = process.env.MONGO_URI;

if (!MONGO_URI) {
  console.error("MONGO_URI not set in .env");
  process.exit(1);
}

mongoose
  .connect(MONGO_URI)
  .then(() => console.log("MongoDB connected"))
  .catch((err) => {
    console.error("MongoDB connection failed:", err.message);
    process.exit(1);
  });

const HealthCheck = mongoose.model(
  "HealthCheck",
  new mongoose.Schema({
    ts: { type: Date, default: Date.now },
    note: String,
  })
);

app.use(express.json());
app.use(cors());

app.get("/health", (req, res) => {
  res.json({ ok: true, service: "express" });
});

// Routes
app.use("/api/auth", authRouter);
app.use("/api", sessionRouter);
app.use("/api", chatRouter);

app.get("/api/db-ping", async (req, res) => {
  try {
    const doc = await HealthCheck.create({ note: "ping" });
    res.json({ ok: true, doc });
  } catch (err) {
    res.status(503).json({
      ok: false,
      error: "mongo write failed",
      detail: err.message,
    });
  }
});

app.get("/api/db-inspect", async (req, res) => {
  try {
    const admin = mongoose.connection.db.admin();
    const dbList = await admin.listDatabases();
    const currentDb = mongoose.connection.db.databaseName;
    const collections = await mongoose.connection.db.listCollections().toArray();
    const healthCount = await HealthCheck.countDocuments();
    res.json({
      ok: true,
      connectedTo: currentDb,
      allDatabases: dbList.databases.map((d) => d.name),
      collectionsInCurrentDb: collections.map((c) => c.name),
      healthCheckDocCount: healthCount,
    });
  } catch (err) {
    res.status(503).json({ ok: false, error: err.message });
  }
});

app.get("/api/ping", async (req, res) => {
  try {
    const response = await fetch(`${FASTAPI_URL}/health`);
    if (!response.ok) {
      return res.status(502).json({
        ok: false,
        error: `fastapi responded with ${response.status}`,
      });
    }
    const data = await response.json();
    res.json({ ok: true, fastapi: data });
  } catch (err) {
    res.status(503).json({
      ok: false,
      error: "fastapi unreachable",
      detail: err.message,
    });
  }
});

app.listen(PORT, () => {
  console.log(`Express listening on http://localhost:${PORT}`);
});

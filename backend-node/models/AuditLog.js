import mongoose from "mongoose";

const auditLogSchema = new mongoose.Schema({
  userId: { type: mongoose.Schema.Types.ObjectId, ref: "User", index: true },
  action: { type: String, required: true, index: true },
  resource: { type: String },
  resourceId: { type: String },
  metadata: { type: mongoose.Schema.Types.Mixed },
  ip: { type: String },
  timestamp: { type: Date, default: Date.now, index: true },
});

// Auto-expire after 1 year — compliance-friendly retention
auditLogSchema.index(
  { timestamp: 1 },
  { expireAfterSeconds: 365 * 24 * 60 * 60 }
);

export default mongoose.model("AuditLog", auditLogSchema);

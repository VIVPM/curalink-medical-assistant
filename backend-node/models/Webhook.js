import mongoose from "mongoose";

const webhookSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "User",
    required: true,
    index: true,
  },
  url: { type: String, required: true },
  secret: { type: String, required: true },
  events: {
    type: [String],
    enum: ["job.completed", "job.failed"],
    default: ["job.completed"],
  },
  active: { type: Boolean, default: true },
  createdAt: { type: Date, default: Date.now },
});

export default mongoose.model("Webhook", webhookSchema);

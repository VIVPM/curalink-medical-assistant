// MongoDB schema for user and assistant messages.

import mongoose from "mongoose";

const messageSchema = new mongoose.Schema(
  {
    sessionId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Session",
      required: true,
      index: true,
    },
    role: {
      type: String,
      enum: ["user", "assistant"],
      required: true,
    },
    content: { type: String, required: true },
    ownKey: { type: Boolean, default: false },
    structuredResponse: { type: mongoose.Schema.Types.Mixed, default: null },
    pipelineMeta: { type: mongoose.Schema.Types.Mixed, default: null },
  },
  { timestamps: true }
);


messageSchema.index({ sessionId: 1, createdAt: 1 });


messageSchema.index({ createdAt: 1 }, { expireAfterSeconds: 90 * 24 * 60 * 60 });

export default mongoose.model("Message", messageSchema);

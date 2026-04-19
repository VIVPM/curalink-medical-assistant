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
    structuredResponse: { type: mongoose.Schema.Types.Mixed, default: null },
    pipelineMeta: { type: mongoose.Schema.Types.Mixed, default: null },
  },
  { timestamps: true }
);

// Compound index for fetching chat history in order
messageSchema.index({ sessionId: 1, createdAt: 1 });

export default mongoose.model("Message", messageSchema);

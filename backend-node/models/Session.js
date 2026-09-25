// MongoDB schema for de-identified research sessions.

import mongoose from "mongoose";

const sessionSchema = new mongoose.Schema(
  {
    userId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      required: true,
      index: true,
    },
    staticContext: {
      disease: { type: String, required: true },
      intent: { type: String, default: "" },
      location: { type: String, default: "" },
    },
    title: { type: String, default: "" },
    messageCount: { type: Number, default: 0 },
  },
  { timestamps: true }
);

sessionSchema.index({ updatedAt: 1 }, { expireAfterSeconds: 90 * 24 * 60 * 60 });

sessionSchema.pre("save", function () {
  if (!this.title) {
    const disease = this.staticContext?.disease || "Untitled";
    const intent = this.staticContext?.intent;
    this.title = intent ? `${disease} — ${intent}` : disease;
  }
});

export default mongoose.model("Session", sessionSchema);

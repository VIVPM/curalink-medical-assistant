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
      patientName: { type: String, default: "" },
    },
    title: { type: String, default: "" },
    messageCount: { type: Number, default: 0 },
  },
  { timestamps: true }
);

// Denormalized title for sidebar: "Parkinson's — DBS" or just disease
sessionSchema.pre("save", function () {
  if (!this.title) {
    const disease = this.staticContext?.disease || "Untitled";
    const intent = this.staticContext?.intent;
    this.title = intent ? `${disease} — ${intent}` : disease;
  }
});

export default mongoose.model("Session", sessionSchema);

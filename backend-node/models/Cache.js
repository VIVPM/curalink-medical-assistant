import mongoose from "mongoose";

const cacheSchema = new mongoose.Schema(
  {
    key: { type: String, required: true, unique: true, index: true },
    response: { type: mongoose.Schema.Types.Mixed, required: true },
    expiresAt: { type: Date, required: true, index: { expires: 0 } },
  },
  { timestamps: true }
);

export default mongoose.model("Cache", cacheSchema);

import mongoose from "mongoose";
import bcrypt from "bcryptjs";

const userSchema = new mongoose.Schema(
  {
    name: { type: String, required: true, trim: true },
    email: {
      type: String,
      required: true,
      unique: true,
      lowercase: true,
      trim: true,
    },
    password: { type: String, required: true, minlength: 8 },
    // ponytail: credits field kept for backward compat reads; daily quota is now
    // ponytail: credits field kept for backward compat; daily quota is window-based
    // (count messages today, cap from DAILY_MESSAGE_CAP env). No decrement, no
    // nightly reset job — at midnight the count is 0 again automatically.
    credits: { type: Number, default: () => Number(process.env.DAILY_MESSAGE_CAP) || 5 },
  },
  { timestamps: true }
);

userSchema.pre("save", async function () {
  if (!this.isModified("password")) return;
  this.password = await bcrypt.hash(this.password, 10);
});

userSchema.methods.comparePassword = function (candidate) {
  return bcrypt.compare(candidate, this.password);
};

export default mongoose.model("User", userSchema);

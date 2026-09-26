import test from "node:test";
import assert from "node:assert/strict";
import { cacheKey } from "./chat.js";

const history = [{ role: "user", content: "Earlier question" }];
const key = (user, location) =>
  cacheKey(user, "Parkinson's", "DBS", location, "Latest evidence?", history);

test("query cache is isolated by user and location", () => {
  assert.notEqual(key("user-a", "Toronto"), key("user-b", "Toronto"));
  assert.notEqual(key("user-a", "Toronto"), key("user-a", "Boston"));
  assert.equal(key("USER-A", " Toronto "), key("user-a", "toronto"));
});

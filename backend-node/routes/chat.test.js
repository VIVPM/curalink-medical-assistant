// Tests chat cache isolation and urgent-use routing.

import test from "node:test";
import assert from "node:assert/strict";
import { cacheKey, isEmergencyMessage } from "./chat.js";

const history = [{ role: "user", content: "Earlier question" }];
const key = (user, location) =>
  cacheKey(user, "Parkinson's", "DBS", location, "Latest evidence?", history);

test("query cache is isolated by user and location", () => {
  assert.match(key("user-a", "Toronto"), /^query:user-a:/);
  assert.notEqual(key("user-a", "Toronto"), key("user-b", "Toronto"));
  assert.notEqual(key("user-a", "Toronto"), key("user-a", "Boston"));
  assert.equal(key("USER-A", " Toronto "), key("user-a", "toronto"));
});

test("query cache includes prior evidence", () => {
  const build = (snippet) => cacheKey("user-a", "Parkinson's", "DBS", "Toronto", "Explain that", [{
    role: "assistant", content: "Same overview", structuredResponse: {
      insights: [{ finding: "Same finding", source_details: [{ url: "https://example.org/study", snippet }] }],
    },
  }]);
  assert.notEqual(build("Evidence A"), build("Evidence B"));
  assert.match(build("Evidence A"), /^query:user-a:[a-f0-9]{64}$/);
});

test("urgent messages are diverted from the research pipeline", () => {
  assert.equal(isEmergencyMessage("I cannot breathe"), true);
  assert.equal(isEmergencyMessage("I took an overdose"), true);
  assert.equal(isEmergencyMessage("What does research say about asthma treatment?"), false);
});

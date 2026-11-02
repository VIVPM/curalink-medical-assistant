import test from 'node:test';
import assert from 'node:assert/strict';
import { retryAfterSeconds } from './polling.mjs';

test('polling honors numeric and HTTP-date Retry-After', () => {
  assert.equal(retryAfterSeconds('30'), 30);
  const now = Date.parse('2026-11-02T10:00:00Z');
  assert.equal(retryAfterSeconds('Mon, 02 Nov 2026 10:01:00 GMT', now), 60);
});

test('missing, invalid or expired retry headers back off safely', () => {
  for (const value of [undefined, '', 'invalid', '-1', '0', 'Infinity']) {
    assert.equal(retryAfterSeconds(value), 5);
  }
});

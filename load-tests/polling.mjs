// Retry-After can be seconds or an HTTP date; never retry immediately on bad input.
export function retryAfterSeconds(header, now = Date.now()) {
  if (!header) return 5;
  const seconds = Number(header);
  const delay = Number.isFinite(seconds) ? seconds : (Date.parse(header) - now) / 1000;
  return Number.isFinite(delay) && delay > 0 ? delay : 5;
}

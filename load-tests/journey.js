import http from 'k6/http';
import { sleep } from 'k6';
import { Counter, Rate, Trend } from 'k6/metrics';
import { retryAfterSeconds } from './polling.mjs';

const base = (__ENV.BASE_URL || 'http://127.0.0.1:4000').replace(/\/$/, '');
if (!/^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/.test(base)) {
  throw new Error('Only a local test API is supported; do not target production.');
}
const accounts = JSON.parse(open(__ENV.ACCOUNTS));
const vus = Number(__ENV.VUS || 1);
const peak = Number(__ENV.PEAK || vus);
if (!Number.isInteger(vus) || !Number.isInteger(peak) || vus < 1 || peak < vus || peak > 150) {
  throw new Error('Require integer 1 <= VUS <= PEAK <= 150.');
}
if (accounts.length < peak || new Set(accounts.map(a => a.email)).size !== accounts.length) {
  throw new Error('Provide a distinct test account for every peak virtual user.');
}
if (!/^load-[a-f0-9]{32}$/.test(__ENV.RUN_TAG || '')) throw new Error('Use run.py to supply a unique run tag.');
const jobs = __ENV.JOBS === '1';
if (jobs && __ENV.ALLOW_MODEL_CALLS !== 'YES' && __ENV.STUBBED !== '1') {
  throw new Error('Job mode requires ALLOW_MODEL_CALLS=YES; jobs can incur provider costs.');
}
const names = ['login', 'sessions', 'session_create', 'session_read', 'session_delete', 'submit', 'job_status'];
const latency = Object.fromEntries(names.map(n => [n, new Trend(`api_${n}`, true)]));
const errors = Object.fromEntries(names.map(n => [n, new Rate(`err_${n}`)]));
const journeyErrors = new Rate('journey_errors');
const created = new Counter('jobs_created');
const succeeded = new Counter('jobs_succeeded');
const failed = new Counter('jobs_failed');
const unfinished = new Counter('jobs_unfinished');
const completion = new Trend('job_completion', true);
const queueWait = new Trend('queue_wait', true);
const ramp = __ENV.RAMP || '30s';
const hold = __ENV.HOLD || '180s';
export const options = {
  scenarios: { users: {
    executor: 'ramping-vus', startVUs: 0,
    stages: __ENV.SPIKE === '1' ? [
      { duration: ramp, target: vus }, { duration: '60s', target: vus },
      { duration: '10s', target: peak }, { duration: '60s', target: peak },
      { duration: '10s', target: vus }, { duration: hold, target: vus },
    ] : [{ duration: ramp, target: vus }, { duration: hold, target: vus }],
    gracefulRampDown: '30s', gracefulStop: '240s',
  } },
  summaryTrendStats: ['p(50)', 'p(95)', 'p(99)', 'avg', 'max', 'count'],
  thresholds: {
    journey_errors: ['rate<0.01'],
    api_sessions: ['p(95)<6000'],
  },
};
function request(name, method, path, body, token, expected) {
  const response = http.request(method, `${base}${path}`, body === null ? null : JSON.stringify(body), {
    timeout: '60s', tags: { name },
    headers: { 'Content-Type': 'application/json', ...(token ? { Authorization: `Bearer ${token}` } : {}) },
  });
  const bad = response.status !== expected;
  latency[name].add(response.timings.duration);
  errors[name].add(bad);
  journeyErrors.add(bad);
  if (bad) {
    if (name === 'job_status' && (response.status === 429 || response.status === 503)) {
      return { retryAfterSeconds: retryAfterSeconds(response.headers['Retry-After']) };
    }
    return null;
  }
  try { return response.json(); } catch (_) { journeyErrors.add(true); return null; }
}
// Each VU owns its session; deletion never touches pre-existing session IDs.
let token;
let session;
let submitted = false;
export default function () {
  const account = accounts[__VU - 1];
  if (!token) {
    const login = request('login', 'POST', '/api/auth/login', {
      email: account.email, password: account.password,
    }, null, 200);
    if (!login || !login.token) { sleep(5); return; }
    token = login.token;
  }
  if (!session) {
    const result = request('session_create', 'POST', '/api/session', {
      disease: 'prediabetes', intent: __ENV.RUN_TAG, location: '',
    }, token, 201);
    session = result && result.session && result.session._id;
    if (!session) { sleep(5); return; }
  }
  request('sessions', 'GET', '/api/sessions', null, token, 200);
  request('session_read', 'GET', `/api/session/${session}`, null, token, 200);
  if (jobs && !submitted) {
    // Never retry an ambiguous submission: a timeout can still have created a job.
    submitted = true;
    const start = Date.now();
    const result = request('submit', 'POST', '/api/jobs', {
      static: { disease: 'prediabetes', intent: __ENV.RUN_TAG, location: '' },
      dynamic: { chatHistory: [] }, current: { userMessage: 'What does prediabetes mean?' },
    }, token, 202);
    if (result && result.job_id) {
      created.add(1);
      let terminal = false;
      while (Date.now() - start < 180000) {
        const status = request('job_status', 'GET', `/api/jobs/${result.job_id}`, null, token, 200);
        if (status && ['completed', 'failed', 'cancelled'].includes(status.state)) {
          terminal = true;
          if (status.state === 'completed') {
            succeeded.add(1); completion.add(Date.now() - start);
            if (status.result && status.result.load_test) queueWait.add(status.result.load_test.queue_wait_ms);
          }
          else { failed.add(1); journeyErrors.add(true); }
          break;
        }
        sleep(Math.min(status && status.retryAfterSeconds || 2, Math.max(0, (180000 - (Date.now() - start)) / 1000)));
      }
      if (!terminal) { unfinished.add(1); journeyErrors.add(true); }
    } else { journeyErrors.add(true); }
  }
  sleep(5 + Math.random() * 5);
}
export function handleSummary(data) {
  return { [__ENV.SUMMARY || 'summary.json']: JSON.stringify(data, null, 2) };
}

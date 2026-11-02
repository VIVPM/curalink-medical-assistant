"""Local test-only gateway for three independent 20-slot simulated job processes."""
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
PORTS = (8022, 8023, 8024)


def aggregate_stats(rows):
    result: dict[str, object] = {key: sum(row[key] for row in rows)
              for key in ('created', 'rejected', 'completed', 'failed', 'pending')}
    result['queue_wait_ms'] = [value for row in rows for value in row['queue_wait_ms']]
    result['processes'] = rows
    return result


class Routing:
    def __init__(self):
        self.lock = threading.Lock()
        self.next = 0
        self.owners = {}

    def choose(self, method, path):
        with self.lock:
            if method == 'POST' and path == '/jobs':
                port = PORTS[self.next % len(PORTS)]
                self.next += 1
                return port
            if path.startswith('/jobs/'):
                return self.owners.get(path.split('/')[2])
            return PORTS[0]

    def remember(self, job_id, port):
        with self.lock:
            self.owners[job_id] = port


def main():
    from dotenv import dotenv_values
    secret = os.environ.get('INTERNAL_API_KEY') or dotenv_values(
        ROOT.parent / 'backend-python/.env')['INTERNAL_API_KEY']
    assert isinstance(secret, str) and secret, 'INTERNAL_API_KEY is required'
    routing = Routing()
    children = []

    def fetch(port, path, method='GET', body=None, headers=None):
        request = Request(f'http://127.0.0.1:{port}{path}', data=body,
                          method=method, headers=headers or {'X-Internal-API-Key': secret})
        try:
            response = urlopen(request, timeout=60)
        except HTTPError as error:
            response = error
        with response:
            return response.status, dict(response.headers), response.read()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def send(self, status, body, headers=None):
            self.send_response(status)
            for name, value in (headers or {}).items():
                if name.lower() in {'content-type', 'retry-after'}:
                    self.send_header(name, value)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def handle_request(self):
            try:
                if self.path.startswith('/load-test/'):
                    if not hmac.compare_digest(self.headers.get('X-Internal-API-Key', ''), secret):
                        self.send(401, b'{"error":"unauthorized"}')
                        return
                    rows = []
                    for port in PORTS:
                        status, _, body = fetch(port, self.path)
                        if status != 200:
                            raise RuntimeError('A worker process is unavailable')
                        rows.append(json.loads(body))
                    if self.path == '/load-test/config':
                        assert len({row['pid'] for row in rows}) == 3
                        assert all(row['workers'] == 20 and row['simulated'] for row in rows)
                        result = {'simulated': True, 'delay_seconds': rows[0]['delay_seconds'],
                                  'workers': 60, 'process_count': 3, 'workers_per_process': 20,
                                  'queue_size_per_process': rows[0]['queue_size'], 'processes': rows}
                    elif self.path.startswith('/load-test/stats?'):
                        result = aggregate_stats(rows)
                    else:
                        self.send(404, b'{"error":"not found"}')
                        return
                    self.send(200, json.dumps(result).encode(), {'Content-Type': 'application/json'})
                    return
                port = routing.choose(self.command, self.path)
                if port is None:
                    self.send(404, b'{"error":"job not found"}')
                    return
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 <= size <= 1024 * 1024:
                    self.send(413, b'{"error":"request too large"}')
                    return
                body = self.rfile.read(size) if size else None
                headers = {key: value for key, value in self.headers.items()
                           if key.lower() not in {'host', 'connection', 'content-length'}}
                status, response_headers, data = fetch(port, self.path, self.command, body, headers)
                if self.command == 'POST' and self.path == '/jobs' and status == 202:
                    routing.remember(json.loads(data)['job_id'], port)
                self.send(status, data, response_headers)
            except Exception as error:
                print('Gateway request failed:', type(error).__name__, flush=True)
                self.send(503, b'{"error":"test worker unavailable"}', {'Retry-After': '2'})

        do_GET = handle_request
        do_POST = handle_request
        do_DELETE = handle_request

    try:
        env = {**os.environ, 'JOB_QUEUE_SIZE': '50'}
        for port in PORTS:
            children.append(subprocess.Popen([sys.executable, str(ROOT / 'stub_server.py'),
                                              '--workers', '20', '--port', str(port)], env=env))
        deadline = time.monotonic() + 120
        for port, child in zip(PORTS, children):
            while True:
                if child.poll() is not None:
                    raise RuntimeError('Worker startup failed')
                try:
                    status, _, data = fetch(port, '/load-test/config')
                    if status == 200 and json.loads(data)['workers'] == 20:
                        break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError('Worker startup timed out')
                time.sleep(.5)
        print('READY: 3 independent Python processes x 20 slots; gateway 127.0.0.1:8021', flush=True)
        ThreadingHTTPServer(('127.0.0.1', 8021), Handler).serve_forever()
    finally:
        for child in children:
            child.terminate()
        for child in children:
            child.wait(timeout=15)


if __name__ == '__main__':
    main()

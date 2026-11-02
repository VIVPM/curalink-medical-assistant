"""Print README-ready latency tables from k6 summaries; no network calls."""
import json
from pathlib import Path
import sys


def table(paths):
    print('| Users | Requests | HTTP errors | Login p50/p95/p99 (s) | Sessions p50/p95/p99 (s) | Submit p50/p95/p99 (s) |')
    print('|---:|---:|---:|---|---|---|')
    for path in paths:
        path = Path(path)
        run = json.loads((path.parent / 'run.json').read_text(encoding='utf-8'))
        metrics = json.loads(path.read_text(encoding='utf-8'))['metrics']
        def values(name):
            return metrics.get(name, {}).get('values', {})
        def percentiles(name):
            data = values(name)
            return '/'.join(f'{data[key] / 1000:.2f}' if key in data else '—'
                            for key in ('p(50)', 'p(95)', 'p(99)'))
        rate = values('http_req_failed').get('rate')
        errors = '—' if rate is None else f'{rate * 100:.2f}%'
        print(f"| {run['users']} | {values('http_reqs').get('count', 0)} | {errors} | "
              f"{percentiles('api_login')} | {percentiles('api_sessions')} | {percentiles('api_submit')} |")
    print('\n| Users | Jobs created | Succeeded | Failed | Job IDs received | Expected | Queue wait p50/p95/p99 (s) |')
    print('|---:|---:|---:|---:|---:|---:|---|')
    for path in paths:
        path = Path(path)
        server_path = path.parent / 'server.json'
        if not server_path.exists():
            continue
        run = json.loads((path.parent / 'run.json').read_text(encoding='utf-8'))
        server = json.loads(server_path.read_text(encoding='utf-8'))
        metrics = json.loads(path.read_text(encoding='utf-8'))['metrics']
        received = metrics.get('jobs_created', {}).get('values', {}).get('count', 0)
        waits = sorted(server['queue_wait_ms'])
        def percentile(p):
            if not waits:
                return '—'
            index = (len(waits) - 1) * p
            low = int(index)
            high = min(low + 1, len(waits) - 1)
            return f'{(waits[low] + (waits[high] - waits[low]) * (index - low)) / 1000:.2f}'
        wait = '/'.join(percentile(p) for p in (0.5, 0.95, 0.99))
        print(f"| {run['users']} | {server['created']} | {server['completed']} | {server['failed']} | {received} | {run['users']} | {wait} |")
    print('\nJob counts and queue wait are server-observed after drain; completion polling may time out sooner. Spike rows aggregate the entire run.')


if __name__ == '__main__':
    table(sys.argv[1:])

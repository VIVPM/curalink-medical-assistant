"""Run local k6 against dedicated accounts; clean only this run's sessions."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from urllib.request import Request, urlopen
import uuid

ROOT = Path(__file__).resolve().parent


def api(base, path, method='GET', body=None, token=None):
    headers = {'Content-Type': 'application/json'}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    request = Request(base + path, data=None if body is None else json.dumps(body).encode(),
                      headers=headers, method=method)
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def cleanup(base, accounts, tag):
    errors = 0
    for account in accounts:
        try:
            token = api(base, '/api/auth/login', 'POST', account)['token']
            sessions = api(base, '/api/sessions', token=token)['sessions']
            for session in sessions:
                if session['title'] == f'prediabetes — {tag}':
                    api(base, f"/api/session/{session['_id']}", 'DELETE', token=token)
        except Exception as error:
            errors += 1
            print(f'Cleanup failed ({type(error).__name__}); retry with --cleanup-tag {tag}')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accounts', type=Path, required=True)
    parser.add_argument('--base', default='http://127.0.0.1:4000')
    parser.add_argument('--users', type=int, default=1)
    parser.add_argument('--peak', type=int)
    parser.add_argument('--spike', action='store_true')
    parser.add_argument('--jobs', action='store_true')
    parser.add_argument('--allow-model-calls', action='store_true')
    parser.add_argument('--stubbed', action='store_true')
    parser.add_argument('--cleanup-tag')
    args = parser.parse_args()
    base = args.base.rstrip('/')
    if not re.fullmatch(r'http://(?:localhost|127\.0\.0\.1)(?::\d+)?', base):
        parser.error('Only a local test API is supported')
    peak = args.peak or args.users
    if not 1 <= args.users <= peak <= 150:
        parser.error('Require 1 <= users <= peak <= 150')
    accounts = json.loads(args.accounts.read_text(encoding='utf-8'))
    if not isinstance(accounts, list) or len(accounts) < peak:
        parser.error('Provide at least one dedicated account per peak VU')
    if any(set(a) != {'email', 'password'} or not a['email'] or not a['password'] for a in accounts):
        parser.error('Every account must contain only non-empty email and password fields')
    if len({a['email'] for a in accounts}) != len(accounts):
        parser.error('Accounts must be distinct')
    if args.cleanup_tag:
        if not re.fullmatch(r'load-[a-f0-9]{32}', args.cleanup_tag):
            parser.error('Invalid cleanup tag')
        return bool(cleanup(base, accounts, args.cleanup_tag))
    if args.stubbed:
        from dotenv import dotenv_values
        config = dotenv_values(ROOT.parent / 'backend-python' / '.env')
        request = Request('http://127.0.0.1:8021/load-test/config', headers={
            'X-Internal-API-Key': os.environ.get('INTERNAL_API_KEY') or config['INTERNAL_API_KEY']})
        with urlopen(request, timeout=10) as response:
            stub = json.load(response)
        if not stub.get('simulated') or base != 'http://127.0.0.1:4021':
            parser.error('Stub mode requires the dedicated local API at port 4021')
        if stub.get('process_count') != 3 or stub.get('workers_per_process') != 20:
            parser.error('Capacity testing requires three independent processes with 20 slots each')
    accounts = accounts[:peak]
    if args.jobs and not (args.allow_model_calls or args.stubbed):
        parser.error('--jobs requires --allow-model-calls or verified --stubbed')
    k6 = shutil.which('k6')
    if not k6:
        parser.error('Install k6 before running')
    tag = 'load-' + uuid.uuid4().hex
    folder = ROOT / 'results' / tag
    folder.mkdir(parents=True)
    env = {**os.environ, 'BASE_URL': base, 'ACCOUNTS': str(args.accounts.resolve()),
           'VUS': str(args.users), 'PEAK': str(peak), 'SPIKE': str(int(args.spike)),
           'JOBS': str(int(args.jobs)), 'STUBBED': str(int(args.stubbed)), 'ALLOW_MODEL_CALLS': 'YES' if args.allow_model_calls else '',
           'RUN_TAG': tag, 'SUMMARY': str(folder / 'summary.json')}
    (folder / 'run.json').write_text(json.dumps({'tag': tag, 'base': base, 'users': args.users,
        'peak': peak, 'spike': args.spike, 'live_jobs': args.jobs and not args.stubbed,
        'simulated_ai': args.stubbed, 'stub_config': stub if args.stubbed else None}, indent=2), encoding='utf-8')
    print(f'Run {tag}; results: {folder}', flush=True)
    code = 1
    try:
        code = subprocess.run([k6, 'run', str(ROOT / 'journey.js')], env=env, check=False).returncode
    finally:
        if cleanup(base, accounts, tag):
            code = 1
    return code


if __name__ == '__main__':
    raise SystemExit(main())

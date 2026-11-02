"""Sequential 25–150 user capacity runs, draining the stub workers between levels."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import Request, urlopen

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent
config = dotenv_values(ROOT.parent / 'backend-python' / '.env')
secret = os.environ.get('INTERNAL_API_KEY') or config['INTERNAL_API_KEY']


def stats(tag):
    request = Request(f'http://127.0.0.1:8021/load-test/stats?tag={tag}',
                      headers={'X-Internal-API-Key': secret})
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def main():
    manifest = ROOT / 'results' / 'capacity-suite.json'
    records = json.loads(manifest.read_text(encoding='utf-8')) if '--resume' in sys.argv else []
    completed = set()
    for record in records:
        folder = Path(record['folder'])
        if not (folder / 'summary.json').exists() or not (folder / 'server.json').exists():
            raise RuntimeError('Saved level is incomplete; refusing to skip it')
        completed.add(record['users'])
    env = {**os.environ, 'RAMP': '30s', 'HOLD': '180s'}
    for users in (25, 50, 75, 100, 150):
        if users in completed:
            continue
        before = set((ROOT / 'results').glob('load-*'))
        print(f'START {users} users', flush=True)
        result = subprocess.run([sys.executable, str(ROOT / 'run.py'), '--accounts',
            str(ROOT / 'accounts-capacity.json'), '--base', 'http://127.0.0.1:4021',
            '--users', str(users), '--jobs', '--stubbed'], env=env, check=False)
        folders = set((ROOT / 'results').glob('load-*')) - before
        if len(folders) != 1:
            raise RuntimeError('Expected one run folder; refusing ambiguous results')
        folder = folders.pop()
        if not (folder / 'summary.json').exists():
            raise RuntimeError('k6 did not produce a summary; stopping suite')
        deadline = time.monotonic() + 900
        while True:
            data = stats(folder.name)
            if not data['pending']:
                break
            if time.monotonic() > deadline:
                raise RuntimeError('Workers did not drain; stopping before next level')
            time.sleep(2)
        (folder / 'server.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
        records.append({'users': users, 'folder': str(folder), 'exit_code': result.returncode})
        (ROOT / 'results' / 'capacity-suite.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
        print(f"DONE {users}: exit={result.returncode}; jobs={data['completed']}/{data['created']}; failed={data['failed']}", flush=True)
        # A threshold failure is a measured result, not a reason to hide higher levels.
        # Halt on infrastructure/cleanup failure rather than continuing with bad fixtures.
        if result.returncode not in (0, 99):
            raise RuntimeError('Runner/cleanup failed; fix before continuing')
    print('CAPACITY SUITE COMPLETE', flush=True)


if __name__ == '__main__':
    main()

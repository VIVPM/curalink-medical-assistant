"""Offline harness checks. No services, account files or k6 required."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import report
import run


class HarnessTests(unittest.TestCase):
    def test_cleanup_only_deletes_exact_run_title(self):
        tag = 'load-' + 'a' * 32
        responses = [{'token': 'test'}, {'sessions': [
            {'_id': 'own', 'title': f'prediabetes — {tag}'},
            {'_id': 'other', 'title': 'prediabetes — someone else'},
        ]}, {'ok': True}]
        with patch.object(run, 'api', side_effect=responses) as api:
            self.assertEqual(run.cleanup('http://localhost:4000', [{'email': 'test', 'password': 'test'}], tag), 0)
            self.assertEqual(api.call_count, 3)
            self.assertEqual(api.call_args.args[1:3], ('/api/session/own', 'DELETE'))

    def test_rejected_pending_record_does_not_block_drain(self):
        from types import SimpleNamespace
        from load_stats import summarize_jobs
        jobs = [SimpleNamespace(request_data={'_load_accepted': False}, state='pending', result=None),
                SimpleNamespace(request_data={'_load_accepted': True}, state='completed',
                                result={'load_test': {'queue_wait_ms': 12}})]
        result = summarize_jobs(jobs)
        self.assertEqual((result['created'], result['rejected'], result['completed'], result['pending']),
                         (1, 1, 1, 0))
        self.assertEqual(result['queue_wait_ms'], [12])

    def test_cleanup_failure_is_visible(self):
        with patch.object(run, 'api', side_effect=TimeoutError), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run.cleanup('http://localhost:4000', [{}], 'test'), 1)

    def test_report_converts_milliseconds_and_marks_missing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'run.json').write_text(json.dumps({'users': 1}), encoding='utf-8')
            summary = root / 'summary.json'
            summary.write_text(json.dumps({'metrics': {
                'api_login': {'values': {'p(50)': 1500, 'p(95)': 2000, 'p(99)': 2500}},
            }}), encoding='utf-8')
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                report.table([summary])
            self.assertIn('1.50/2.00/2.50', output.getvalue())
            self.assertIn('—/—/—', output.getvalue())


if __name__ == '__main__':
    unittest.main()

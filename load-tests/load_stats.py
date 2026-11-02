"""Count accepted work separately from job records created before queue rejection."""


def summarize_jobs(jobs):
    accepted = [j for j in jobs if j.request_data.get('_load_accepted') is True]
    return {'created': len(accepted), 'rejected': len(jobs) - len(accepted),
            'completed': sum(j.state == 'completed' for j in accepted),
            'failed': sum(j.state == 'failed' for j in accepted),
            'pending': sum(j.state in {'pending', 'running', 'streaming'} for j in accepted),
            'queue_wait_ms': [j.result['load_test']['queue_wait_ms'] for j in accepted if j.result]}

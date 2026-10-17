"""Compare local serving with saved development predictions; never load labels."""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from graphshield.paths import ROOT, inside_root
from graphshield.pipeline import sha256, write_json
from graphshield.service_store import Store
from graphshield.serving import Bundle, InvestigationService
from graphshield.synthetic import checked_name


def verify(name, bundle_name):
    output = inside_root(Path('artifacts/phase6/verification') / checked_name(name), ROOT)
    if output.exists():
        raise ValueError('Choose a new verification run name')
    output.mkdir(parents=True)
    bundle = Bundle(bundle_name)
    store = Store(output / 'service.sqlite')
    service = InvestigationService(bundle, store)
    source = ROOT / 'artifacts/phase5' / bundle.meta['learned_run']
    world = bundle.meta['default_world']
    predictions = {}
    for line in (source / 'predictions.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row['world_id'] == world:
            predictions[row['candidate_id']] = row['scores']
    alerts = [json.loads(line) for line in (source / 'alerts.jsonl').read_text().splitlines()]
    results = []
    start = time.perf_counter()
    for model in bundle.meta['models']:
        for policy in ('tuned_threshold', 'budget_only'):
            job = service.request_replay(world, model, policy)
            final = service.run_replay(job['replay_id'])
            expected = {r['candidate_id'] for r in alerts if r['world_id'] == world and r['model'] == model and r['policy'] == policy}
            actual = [store.case(r['case_id']) for r in store.list_cases(replay_id=job['replay_id'], limit=200)['items']]
            assert {r['candidate_id'] for r in actual} == expected, (model, policy, 'alert parity')
            difference = max((abs(r['score'] - predictions[r['candidate_id']][model]) for r in actual), default=0)
            assert difference < 1e-6, (model, policy, difference)
            for case in actual:
                for rows in case['evidence'].values():
                    assert all(case['window_start'] <= r['step'] < case['cutoff_step'] and r['available_step'] < case['cutoff_step'] for r in rows)
                contribution = case['contributions']
                if contribution['available']:
                    margin = contribution['bias'] + sum(contribution['features'].values())
                    probability = 1 / (1 + math.exp(-margin))
                    assert abs(probability - case['score']) < 1e-6
            revision = store.world(world)['revision']
            assert service.run_replay(job['replay_id']) == final
            assert store.world(world)['revision'] == revision
            assert final['completed_snapshots'] == len(bundle.meta['cutoffs'])
            results.append({'model':model, 'policy':policy, 'snapshots':final['completed_snapshots'],
                            'alerts':len(actual), 'maximum_score_difference':difference,
                            'alert_parity':True, 'temporal_evidence_valid':True, 'retry_unchanged':True})
            print(f'{model} / {policy}: {len(actual)} alerts, parity passed', flush=True)
    report = {'status':'passed', 'bundle_version':bundle.version, 'world_id':world,
              'scope':'Serving parity on the synthetic development comparison network; no new ML evaluation',
              'final_test_evaluated':False, 'ground_truth_loaded':False, 'seconds':time.perf_counter()-start,
              'replays':results, 'source_sha256':{p.name:sha256(p) for p in [
                  ROOT/'src/graphshield/service_store.py', ROOT/'src/graphshield/serving.py',
                  ROOT/'src/graphshield/api.py', Path(__file__)]}}
    write_json(output/'report.json', report)
    print(json.dumps({'status':'passed','replays':len(results),'report':str(output/'report.json')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--bundle-name', default='serving-v1')
    args = parser.parse_args()
    verify(args.run_name, args.bundle_name)

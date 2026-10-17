"""Recheck data integrity and structural contracts without fitting a model."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from graphshield.paths import inside_root  # noqa: E402
from graphshield.pipeline import sha256, write_json  # noqa: E402
from graphshield.synthetic import checked_name  # noqa: E402
from graphshield.synthetic_audit import audit_benchmark  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', required=True)
    args = parser.parse_args()
    reports = inside_root(Path('artifacts/phase3') / checked_name(args.run_name))
    manifest = json.loads((reports / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'complete':
        raise ValueError('Only a completed generation run can be consumed')
    report = audit_benchmark(inside_root(manifest['output_path']), manifest)
    checks = dict(report['checks'])
    checks['saved_report_reproduces'] = report == json.loads((reports / 'report.json').read_text(encoding='utf-8'))
    checks['generator_source_matches_manifest'] = sha256(ROOT / 'src/graphshield/synthetic.py') == manifest['generator_code_sha256']
    checks['audit_source_matches_manifest'] = sha256(ROOT / 'src/graphshield/synthetic_audit.py') == manifest['audit_code_sha256']
    write_json(reports / 'verification.json', checks)
    print(json.dumps(checks, indent=2))
    if not all(checks.values()):
        raise SystemExit('Phase 3 verification failed')


if __name__ == '__main__':
    main()

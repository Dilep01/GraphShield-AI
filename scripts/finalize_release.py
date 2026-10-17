"""Seal or verify the local 0.7.0 release; uses only the Python standard library."""
import argparse
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / 'artifacts/phase7'
RELEASE = PHASE / 'release'
VERSION = '0.7.0'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def contained(base, name):
    path = (base / name).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError(f'Path escapes its root: {name}')
    return path


def check_files(base, files):
    for name, expected in files.items():
        path = contained(base, name)
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f'Missing or changed file: {name}')


def archive_matches(path, expected):
    with zipfile.ZipFile(path) as archive:
        if len(archive.namelist()) != len(expected) or set(archive.namelist()) != set(expected):
            raise ValueError(f'Archive members differ: {path.name}')
        for name, expected_digest in expected.items():
            contained(ROOT, name)
            if hashlib.sha256(archive.read(name)).hexdigest() != expected_digest:
                raise ValueError(f'Archive content changed: {name}')


def verify(release_only=False):
    manifest = read(RELEASE / 'manifest.json')
    check_files(RELEASE, manifest['files'])
    archive_matches(RELEASE / f'graphshield-source-v{VERSION}.zip', manifest['source_files'])
    demo = read(RELEASE / 'demo-data-manifest.json')
    archive_matches(RELEASE / f'graphshield-demo-data-v{VERSION}.zip',
                    {Path(name).as_posix(): value for name, value in demo['files'].items()})
    if not release_only:
        completion = read(PHASE / 'manifest.json')
        if completion['status'] != 'complete':
            raise ValueError('Phase 7 is not marked complete')
        check_files(ROOT, completion['evidence_files'])
        check_files(ROOT, manifest['source_files'])
    print(json.dumps({'status': 'passed', 'release_files': len(manifest['files']),
                      'source_files': len(manifest['source_files']),
                      'scope': 'release only' if release_only else 'release, source and completion evidence'}, indent=2))


def finalize():
    destinations = [RELEASE / 'manifest.json', PHASE / 'manifest.json',
                    RELEASE / f'graphshield-source-v{VERSION}.zip']
    if any(path.exists() for path in destinations):
        raise ValueError('Release already sealed or partially created; preserve it and inspect before rebuilding')
    final = PHASE / 'final-v1'
    verification = read(final / 'verification.json')
    reproduction = read(PHASE / 'reproduction/report.json')
    browser = read(PHASE / 'browser-qa.json')
    status = read(final / 'status.json')
    if (verification['status'] != 'passed' or not all(verification['checks'].values())
            or reproduction['status'] != 'passed' or browser['status'] != 'passed'
            or status['status'] != 'complete'
            or status['results_sha256'] != digest(final / 'results.json')):
        raise ValueError('Required completion evidence is missing or failed')
    suites = ET.parse(PHASE / 'clean-tests.xml').getroot().findall('testsuite')
    counts = {key: sum(int(s.attrib.get(key, 0)) for s in suites)
              for key in ('tests', 'failures', 'errors', 'skipped')}
    if counts != {'tests': 96, 'failures': 0, 'errors': 0, 'skipped': 0}:
        raise ValueError(f'Unexpected test record: {counts}')
    wheel = RELEASE / f'graphshield_ai-{VERSION}-py3-none-any.whl'
    with zipfile.ZipFile(wheel) as archive:
        package_files = [p for p in (ROOT / 'src/graphshield').rglob('*')
                         if p.is_file() and p.suffix in {'.py', '.html', '.css', '.js'}]
        for path in package_files:
            if archive.read(path.relative_to(ROOT / 'src').as_posix()) != path.read_bytes():
                raise ValueError(f'Wheel differs from source: {path}')
    sources = [ROOT / name for name in ('README.md', 'pyproject.toml', 'requirements-lock.txt', '.gitignore')]
    allowed = {'src/graphshield': {'.py', '.html', '.css', '.js'}, 'tests': {'.py'},
               'scripts': {'.py', '.ps1'}, 'config': {'.json'}, 'docs': {'.md'}}
    for folder, suffixes in allowed.items():
        sources.extend(p for p in (ROOT / folder).rglob('*') if p.is_file() and p.suffix in suffixes)
    source_hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(set(sources))}
    with zipfile.ZipFile(destinations[2], 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in source_hashes:
            archive.write(contained(ROOT, name), name)
    shutil.copyfile(ROOT / 'requirements-lock.txt', RELEASE / 'requirements-lock.txt')
    shutil.copyfile(ROOT / 'docs/RUNBOOK.md', RELEASE / 'RUNBOOK.md')
    wheelhouse = list((RELEASE / 'wheelhouse').glob('*.whl'))
    if len(wheelhouse) != 42:
        raise ValueError('Expected the 42 pinned dependency wheels')
    release_files = {p.relative_to(RELEASE).as_posix(): digest(p)
                     for p in sorted(RELEASE.rglob('*')) if p.is_file()}
    write(destinations[0], {'version': VERSION, 'completed_date': '2026-10-17',
          'platform': 'Windows x64 / CPython 3.11 / CPU', 'dependency_wheels': len(wheelhouse),
          'files': release_files, 'source_files': source_hashes,
          'provenance': 'Synthetic development demo; excludes raw PaySim and final-test datasets',
          'validation_boundary': '96 installed-wheel tests and replay reproduction before CSS-only fix; final wheel source equality and browser checks after fix'})
    evidence = [final / name for name in ('protocol.json', 'prediction_seal.json', 'status.json',
                                         'results.json', 'verification.json')]
    evidence += [PHASE / 'clean-tests.xml', PHASE / 'reproduction/report.json',
                 PHASE / 'browser-qa.json', PHASE / 'system-health.jpg',
                 PHASE / 'narrow-layout.jpg', PHASE / 'phone-layout.jpg', RELEASE / 'manifest.json']
    write(destinations[1], {'status': 'complete', 'version': VERSION, 'completed_date': '2026-10-17',
          'scope': 'Planned seven-phase local synthetic research prototype',
          'tests': counts, 'final_verification_checks': len(verification['checks']),
          'reproduced_snapshots': reproduction['reproduced_snapshots'],
          'reproduced_cases': reproduction['reproduced_cases'],
          'known_limits': ['Selected group ranker: 19/24 known-family groups, 38% case precision',
                          'All models: 0/24 withheld relay-chain groups; zero candidate coverage',
                          'No production or real-data performance claim; final labels consumed'],
          'evidence_files': {p.relative_to(ROOT).as_posix(): digest(p) for p in evidence}})
    verify()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--release-only', action='store_true')
    args = parser.parse_args()
    if args.release_only and not args.verify:
        parser.error('--release-only requires --verify')
    verify(args.release_only) if args.verify else finalize()

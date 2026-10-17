"""Package the trusted local synthetic demo, with explicit file allowlists."""
import json
import sys
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from graphshield.paths import inside_root
from graphshield.pipeline import sha256,write_json


def archive(path, files):
    if path.exists():raise ValueError('Archive exists; preserve the release or choose a new destination')
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as out:
        for file in sorted(set(files)):
            file=inside_root(file,ROOT)
            out.write(file,file.relative_to(ROOT).as_posix())


def main():
    output=ROOT/'artifacts/phase7/release'
    output.mkdir(parents=True,exist_ok=True)
    bundle_folder=ROOT/'artifacts/phase6/bundles/serving-v1'
    bundle=json.loads((bundle_folder/'bundle.json').read_text())
    files=[p for p in bundle_folder.rglob('*') if p.is_file()]
    files += [inside_root(c['path'],ROOT) for c in bundle['embedding_cache']]
    files += [inside_root(v['path'],ROOT) for w in bundle['worlds'] for v in w['observed'].values()]
    files += [ROOT/'artifacts/phase6/demo.sqlite']
    archive(output/'graphshield-demo-data-v0.7.0.zip',files)
    write_json(output/'demo-data-manifest.json',{'provenance':'Generated synthetic development observations and trusted local model artifacts',
        'contains_final_test':False,'contains_paysim_raw_data':False,'files':{str(p.relative_to(ROOT)):sha256(p) for p in files}})
    print('Synthetic demo data archive and checksums saved.')


if __name__=='__main__':main()

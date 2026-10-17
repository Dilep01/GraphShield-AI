"""Download the public Kaggle archive into this project, without credentials."""

import hashlib
import json
import shutil
import sys
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from graphshield.paths import inside_root  # noqa: E402

URL = "https://www.kaggle.com/api/v1/datasets/download/ealaxi/paysim1"


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main():
    raw = inside_root("data/raw")
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "paysim1.zip"
    partial = raw / "paysim1.zip.part"
    metadata = raw / "download_manifest.json"
    if not archive.exists():
        request = urllib.request.Request(URL, headers={"User-Agent": "GraphShieldAI/0.1"})
        started = time.monotonic()
        total = 0
        with urllib.request.urlopen(request, timeout=90) as response, partial.open("wb") as output:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                output.write(block)
                total += len(block)
                if total % (25 * 1024 * 1024) == 0:
                    print(f"Downloaded {total // (1024 * 1024)} MiB", flush=True)
        if not zipfile.is_zipfile(partial):
            raise ValueError("Response is not a ZIP archive; download not promoted to dataset.")
        partial.replace(archive)
        print(f"Download finished in {time.monotonic() - started:.1f}s", flush=True)
    with zipfile.ZipFile(archive) as source:
        members = [item for item in source.infolist() if item.filename.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"Expected one CSV, received {len(members)}")
        member = members[0]
        # Discard remote path components. Never extract arbitrary archive paths.
        csv_path = inside_root(raw / Path(member.filename).name)
        expected_size = member.file_size
        if csv_path.exists():
            if not metadata.exists():
                raise ValueError("Existing CSV lacks a download manifest; refusing to overwrite it.")
            saved = json.loads(metadata.read_text(encoding="utf-8"))
            if digest(csv_path) != saved["csv_sha256"]:
                raise ValueError("Existing CSV checksum differs from its manifest.")
        else:
            staged = inside_root(csv_path.with_suffix(".csv.part"))
            with source.open(member) as reader, staged.open("wb") as writer:
                shutil.copyfileobj(reader, writer, length=1024 * 1024)
            if staged.stat().st_size != expected_size:
                raise ValueError("Extracted CSV size mismatch")
            staged.replace(csv_path)
    if not metadata.exists():
        metadata.write_text(json.dumps({
            "dataset_id": "ealaxi/paysim1", "source_url": URL,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "archive_sha256": digest(archive), "csv_sha256": digest(csv_path),
            "csv_path": str(csv_path.relative_to(ROOT)), "csv_bytes": expected_size,
            "provenance": "Original public PaySim synthetic dataset; not real customer records"
        }, indent=2) + "\n", encoding="utf-8")
    print(f"Ready: {csv_path}", flush=True)


if __name__ == "__main__":
    main()

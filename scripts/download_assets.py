#!/usr/bin/env python3
"""Download only locked public assets; verify every file before atomic promotion."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import fcntl
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]


def verify(path, entry):
    if not path.is_file() or path.stat().st_size != entry['size']:
        return False
    sha = hashlib.sha256()
    git = hashlib.sha1(f"blob {entry['size']}\0".encode())
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            sha.update(chunk)
            git.update(chunk)
    return (sha.hexdigest() == entry['sha256'] if entry['sha256']
            else git.hexdigest() == entry['git_blob'])


def fetch(entry, endpoint):
    dest = ROOT / entry['local']
    dest.parent.mkdir(parents=True, exist_ok=True)
    # One process per file, even if metadata and full preparation overlap.
    lock_dir = ROOT / '.cache/download-locks'
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / hashlib.sha256(entry['local'].encode()).hexdigest()).open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return fetch_locked(entry, endpoint, dest)


def fetch_locked(entry, endpoint, dest):
    if verify(dest, entry):
        return 'cached'
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + '.partial')
    prefix = 'datasets/' if entry['repo_type'] == 'dataset' else ''
    url = f"{endpoint}/{prefix}{entry['repo']}/resolve/{entry['revision']}/{quote(entry['path'], safe='/=')}"
    # curl supports resumed transfers and follows Hugging Face's signed LFS redirects.
    for attempt in range(3):
        result = subprocess.run([
            'curl', '--fail', '--location', '--silent', '--show-error',
            '--retry', '8', '--retry-all-errors', '--retry-delay', '5', '--connect-timeout', '20',
            '--max-time', '600', '--continue-at', '-', '--output', str(part), url,
        ], capture_output=True, text=True)
        if result.returncode == 0 and verify(part, entry):
            part.replace(dest)
            return 'downloaded'
        # Range unsupported or a complete but invalid file: retry from the start.
        if part.exists() and (result.returncode == 33 or part.stat().st_size >= entry['size']):
            part.unlink()
    raise RuntimeError(f"Download failed: {entry['path']}: {result.stderr[-500:]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', default=os.environ.get('HF_ENDPOINT', 'https://huggingface.co'))
    parser.add_argument(
        '--lock', type=Path,
        default=ROOT / 'assets/manifests/case118-assets.lock.json',
    )
    parser.add_argument('--scope', choices=['metadata', 'full'], default='full')
    parser.add_argument('--partitions', help='Comma-separated partition IDs; implies metadata plus these partitions')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    lock_path = args.lock.resolve()
    if not lock_path.is_relative_to(ROOT.resolve()):
        raise ValueError('asset lock must be inside the repository')
    entries = json.loads(lock_path.read_text())['files']
    partitions = set(map(int, args.partitions.split(','))) if args.partitions else None
    selected = []
    for entry in entries:
        match = re.search(r'scenario_partition=(\d+)/', entry['path'])
        if match and (partitions is not None and int(match[1]) not in partitions
                      or partitions is None and args.scope == 'metadata'):
            continue
        selected.append(entry)
    # Complete each partition before starting the next, so smoke runs become
    # usable early. Partition numbering is NOT used as a scenario ID mapping.
    def order(entry):
        match = re.search(r'scenario_partition=(\d+)/', entry['path'])
        return (1, int(match[1]), entry['path']) if match else (0, 0, entry['path'])
    selected.sort(key=order)
    print(f"Checking/downloading {len(selected)} files ({sum(e['size'] for e in selected)/1e9:.3f} GB)", flush=True)
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, e, args.endpoint.rstrip('/')): e for e in selected}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception as exc:
                errors.append(str(exc))
                print(str(exc), flush=True)
            if i % 50 == 0 or i == len(selected):
                print(f"Verified {i-len(errors)}/{len(selected)}; failures={len(errors)}", flush=True)
    if errors:
        raise SystemExit(f"{len(errors)} files failed. Re-run the same command to resume.")


if __name__ == '__main__':
    main()

"""Verify and restore the large PR #7 evidence files from a checksummed release asset."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
CATALOG = Path(__file__).with_name('release_assets.json')
REPOSITORY = 'muzhilianhua/SDM374_Project-1'
TAG = 'c-pr7-evidence-2026-11-03'
ARCHIVE_NAME = 'C_PR7_LARGE_EVIDENCE_2026-11-03.zip'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(root, catalog):
    missing = []
    for name, record in catalog['files'].items():
        path = root / name
        if not path.is_file():
            missing.append(name)
        elif path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
            raise ValueError(f'asset differs from the archived source: {name}')
    if missing:
        raise ValueError(f'{len(missing)} assets missing; run assets.py restore')
    return len(catalog['files'])


def restore(root, archive, catalog):
    """Validate every member before writing; never overwrite differing local work."""
    if digest(archive) != catalog['archive_sha256']:
        raise ValueError('release archive checksum mismatch')
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if len(names) != len(set(names)) or set(names) != set(catalog['files']):
            raise ValueError('release archive file list mismatch')
        for name in names:
            relative = PurePosixPath(name)
            target = (root / name).resolve()
            if relative.is_absolute() or '..' in relative.parts or not target.is_relative_to(root.resolve()):
                raise ValueError(f'unsafe asset path: {name}')
            record = catalog['files'][name]
            data = bundle.read(name)
            if len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
                raise ValueError(f'asset checksum mismatch: {name}')
            if target.exists() and (not target.is_file() or digest(target) != record['sha256']):
                raise ValueError(f'refusing to overwrite differing local file: {name}')
        for name in names:
            target = root / name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(name) as source, target.open('wb') as destination:
                    shutil.copyfileobj(source, destination)
    return verify(root, catalog)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('restore', 'verify'))
    parser.add_argument('--archive', type=Path, help='Use a previously downloaded evidence ZIP')
    args = parser.parse_args()
    catalog = json.loads(CATALOG.read_text())
    if args.command == 'verify':
        count = verify(ROOT, catalog)
    elif args.archive:
        count = restore(ROOT, args.archive, catalog)
    else:
        with tempfile.TemporaryDirectory(prefix='sdm-c-evidence-') as directory:
            subprocess.run(['gh', 'release', 'download', TAG, '--repo', REPOSITORY,
                '--pattern', ARCHIVE_NAME, '--dir', directory], check=True)
            count = restore(ROOT, Path(directory) / ARCHIVE_NAME, catalog)
    print(json.dumps({'verified_files': count, 'sha256_verified': True}))


if __name__ == '__main__':
    main()

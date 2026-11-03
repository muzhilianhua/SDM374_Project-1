"""Prevent stale known-point status and verify safe restoration of archived evidence."""
import hashlib
import importlib.util
from pathlib import Path
import zipfile

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'analysis/submission_handoff'/f'{name}.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_old_pending_sheet_cannot_be_presented_as_current():
    package = module('package')
    historical = pd.read_csv(ROOT/'analysis/online_round1/delivery/pending_results.csv')
    with pytest.raises(AssertionError):
        package.validate_status_table(historical)
    historical['status'] = 'unmeasured_not_scheduled'
    for candidate, value in package.known_pool_results().items():
        mask = historical.candidate_id == candidate
        historical.loc[mask, ['status', 'measured_R']] = ['known_do_not_resubmit', value]
    package.validate_status_table(historical)
    historical.loc[historical.candidate_id == 'r01-003', 'measured_R'] = 112.
    with pytest.raises(AssertionError):
        package.validate_status_table(historical)


def archive_fixture(tmp_path):
    path = tmp_path/'asset.zip'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('evidence/raw.txt', b'original evidence')
    catalog = {'archive_sha256':hashlib.sha256(path.read_bytes()).hexdigest(), 'files':{
        'evidence/raw.txt':{'bytes':17,'sha256':hashlib.sha256(b'original evidence').hexdigest()}}}
    return path, catalog


def test_asset_restore_is_verified_and_idempotent(tmp_path):
    assets = module('assets')
    archive, catalog = archive_fixture(tmp_path)
    destination = tmp_path/'checkout'
    assert assets.restore(destination, archive, catalog) == 1
    assert assets.restore(destination, archive, catalog) == 1
    assert (destination/'evidence/raw.txt').read_bytes() == b'original evidence'


def test_asset_restore_preserves_differing_local_work(tmp_path):
    assets = module('assets')
    archive, catalog = archive_fixture(tmp_path)
    destination = tmp_path/'checkout'
    (destination/'evidence').mkdir(parents=True)
    (destination/'evidence/raw.txt').write_bytes(b'local edits')
    with pytest.raises(ValueError, match='refusing to overwrite'):
        assets.restore(destination, archive, catalog)
    assert (destination/'evidence/raw.txt').read_bytes() == b'local edits'


def test_asset_restore_rejects_modified_archive_before_writing(tmp_path):
    assets = module('assets')
    archive, catalog = archive_fixture(tmp_path)
    with archive.open('ab') as f:
        f.write(b'tampered')
    destination = tmp_path/'checkout'
    with pytest.raises(ValueError, match='checksum mismatch'):
        assets.restore(destination, archive, catalog)
    assert not destination.exists()

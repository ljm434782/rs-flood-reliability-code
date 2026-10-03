"""Byte-integrity check of the six copied original archives, not a new source audit."""
import concurrent.futures
import hashlib
import json
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def check(entry):
    path=ROOT/entry['path']
    if not path.is_file() or path.stat().st_size!=entry['size_bytes']:
        raise RuntimeError('Copied archive missing or size mismatch: '+entry['path'])
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(8*1024*1024):
            digest.update(block)
    actual=digest.hexdigest()
    if actual!=entry['sha256']:
        raise RuntimeError('Copied archive SHA256 mismatch: '+entry['path'])
    row={'path':entry['path'],'size_bytes':entry['size_bytes'],'sha256':actual,'status':'MATCH'}
    print(json.dumps(row),flush=True)
    return row


if __name__=='__main__':
    config=json.loads((ROOT/'configs/upgrade_20260929.json').read_text())
    start=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        rows=list(pool.map(check,config['data_source']['raw_archives']))
    (ROOT/'migration_raw_transfer_check.json').write_text(json.dumps(
        {'status':'PASS','archives':rows,'elapsed_seconds':time.monotonic()-start},indent=2)+'\n')
    print('ALL_SIX_ARCHIVE_BYTES_MATCH',flush=True)

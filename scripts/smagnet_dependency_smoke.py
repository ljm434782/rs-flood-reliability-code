"""One synthetic SMAGNet CUDA step after installing its missing import dependency.

Run from the isolated server project root; never opens raster data or starts a fit.
The original runtime lock is retained, and any changed existing package blocks recovery.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
import torch
from scripts.upgrade_train import make_model, _train_epoch


def main():
    before_path = ROOT / 'cuda_environment.before_smagnet_fix.lock.txt'
    after_path = ROOT / 'cuda_environment.after_smagnet_fix.lock.txt'
    before = set(before_path.read_text().splitlines())
    after = set(after_path.read_text().splitlines())
    if before - after:
        raise RuntimeError('Existing runtime packages changed: ' + repr(sorted(before - after)))
    config = json.loads((ROOT / 'configs/upgrade_20260929.json').read_text())
    config['_active_seed'] = 20261013
    config['_active_epoch'] = 1
    torch.set_num_threads(4)
    torch.manual_seed(20261013)
    torch.cuda.manual_seed_all(20261013)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device('cuda')
    batch = (torch.randn(4, 2, 256, 256, device=device),
             torch.randn(4, 12, 256, 256, device=device),
             torch.ones(4, 1, 256, 256, dtype=torch.bool, device=device),
             torch.randint(0, 3, (4, 256, 256), device=device),
             torch.zeros(4, device=device))
    batch[3][:, 0, 0] = 255
    start = time.monotonic()
    model = make_model('SMAGNet', ROOT, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, foreach=False)
    result = _train_epoch(model, 'SMAGNet', [(batch, [f'synthetic-{i}' for i in range(4)])],
                          optimizer, {'class_weights': [1., 2., 3.]}, config, device)
    report = {'status': 'PASS', 'purpose': 'missing dependency recovery check',
              'data_access': 'none; synthetic tensors only', 'model': 'SMAGNet',
              'precision': 'FP32; no AMP; TF32 disabled', 'batch_size': 4,
              'duration_seconds': time.monotonic() - start, 'step': result,
              'added_packages': sorted(after - before), 'changed_existing_packages': [],
              'environment_before_sha256': hashlib.sha256(before_path.read_bytes()).hexdigest(),
              'environment_after_sha256': hashlib.sha256(after_path.read_bytes()).hexdigest()}
    (ROOT / 'smagnet_dependency_smoke.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()

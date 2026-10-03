#!/bin/bash
set -euo pipefail
cd /home/RS_upgrade_20260929
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
mkdir -p outputs/upgrade_20260929/orchestrator
exec 9>server_migration.process.lock
flock -n 9 || { echo "SERVER_PIPELINE_ALREADY_ACTIVE"; exit 2; }
test -f CUDA_ENV_READY
if ! test -f migration_raw_transfer_check.json; then
    env/bin/python -u scripts/verify_transferred_archives.py
fi
if ! test -f cuda_smoke.json; then
    env/bin/python -u scripts/smoke_cuda_migration.py --device cuda --output cuda_smoke.json
fi
if ! test -f migration_artifact_import.json; then
    env/bin/python -u scripts/import_completed_migration.py
fi
env/bin/python -u scripts/upgrade_20260929.py run-all
echo SERVER_MATRIX_COMPLETE

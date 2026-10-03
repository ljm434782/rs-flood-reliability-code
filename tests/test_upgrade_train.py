from pathlib import Path

import pytest
import torch

from flood_reliability.models import SmallUNet
from scripts.upgrade_train import train_synthetic, execute_fit, _enforce_incremental_disk_cap


def _batches():
    torch.manual_seed(100)
    return [[(torch.randn(2, 2, 8, 8), torch.randint(0, 3, (2, 8, 8)))
             for _ in range(2)] for _ in range(3)]


def _model():
    torch.manual_seed(55)
    return SmallUNet(2, 3, 4)


def test_synthetic_train_checkpoint_resume_matches_uninterrupted(tmp_path: Path):
    hashes = {"config": "cfg", "fit": "fit", "code": "code"}
    uninterrupted = _model()
    train_synthetic(uninterrupted, _batches(), run_dir=tmp_path / "full",
                    run_id="synthetic-seed-1", seed=1, max_epochs=3, hashes=hashes)

    resumed = _model()
    train_synthetic(resumed, _batches(), run_dir=tmp_path / "resume",
                    run_id="synthetic-seed-1", seed=1, max_epochs=3,
                    hashes=hashes, stop_after_epoch=1)
    assert (tmp_path / "resume" / "latest.pt").is_file()
    history = train_synthetic(resumed, _batches(), run_dir=tmp_path / "resume",
                              run_id="synthetic-seed-1", seed=1, max_epochs=3,
                              hashes=hashes)
    assert [row["epoch"] for row in history] == [1, 2, 3]
    for name, expected in uninterrupted.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], expected, rtol=0, atol=0)
    assert (tmp_path / "resume" / "epochs.jsonl").read_text().count("\n") == 3


def test_synthetic_resume_rejects_hash_mismatch(tmp_path: Path):
    model = _model()
    train_synthetic(model, _batches(), run_dir=tmp_path / "run", run_id="r", seed=2,
                    max_epochs=3, hashes={"config": "a"}, stop_after_epoch=1)
    with pytest.raises(RuntimeError, match="hash/run mismatch"):
        train_synthetic(_model(), _batches(), run_dir=tmp_path / "run", run_id="r",
                        seed=2, max_epochs=3, hashes={"config": "b"})


def test_synthetic_all_ignore_fails_before_checkpoint(tmp_path: Path):
    batches = [[(torch.zeros(1, 2, 8, 8), torch.full((1, 8, 8), 255, dtype=torch.long))]]
    with pytest.raises(RuntimeError, match="all-Ignore"):
        train_synthetic(_model(), batches, run_dir=tmp_path / "run", run_id="r",
                        seed=3, max_epochs=1, hashes={})
    assert not (tmp_path / "run" / "latest.pt").exists()


def test_production_entry_refuses_before_touching_draft(tmp_path: Path):
    with pytest.raises(RuntimeError, match="not FROZEN"):
        execute_fit(tmp_path, {"status": "DRAFT_NOT_FROZEN",
                               "freeze_required_files": ["G3_FREEZE_MANIFEST.json",
                                                          "G3_INDEPENDENT_REVIEW.json"]},
                    "B2", 20261013, "cpu")


def test_production_disk_reservation_is_bounded_and_uses_existing_volume(tmp_path: Path):
    run_dir = tmp_path / "outputs" / "upgrade_20260929" / "B2_seed_20261013"
    run_dir.mkdir(parents=True)
    model = _model()
    config = {"resources": {"durable_incremental_disk_bytes": 3_000_000_000}}
    reserved = _enforce_incremental_disk_cap(tmp_path, run_dir, model, config)
    assert reserved == sum(p.numel() for p in model.parameters()) * 32 + 64 * 1024 * 1024

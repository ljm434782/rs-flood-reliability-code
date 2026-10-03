import json

import pytest

from scripts import smoke_upgrade_fit


def test_fit_smoke_refuses_draft_before_reader_construction(tmp_path, monkeypatch):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "upgrade_20260929.json").write_text(
        json.dumps({"status": "DRAFT_NOT_FROZEN"}), encoding="utf-8")

    def forbidden_reader(*_args, **_kwargs):
        raise AssertionError("reader must not be constructed for a draft config")

    monkeypatch.setattr(smoke_upgrade_fit, "TrainArchiveReader", forbidden_reader)
    with pytest.raises(RuntimeError, match="requires the protocol config status FROZEN"):
        smoke_upgrade_fit.run_smoke(tmp_path)

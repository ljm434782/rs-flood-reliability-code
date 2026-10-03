from pathlib import Path

import pytest
import torch

from flood_reliability.smagnet_adapter import (
    IGNORE_INDEX,
    make_smagnet_resnet50,
    split_s1_s2,
    two_head_loss,
    validate_batch,
)


ROOT = Path(__file__).resolve().parents[1]
AUTHOR_SOURCE = ROOT / "third_party" / "smagnet" / "smagnet.py"


def synthetic_batch(device: torch.device):
    torch.manual_seed(17)
    x = torch.randn(1, 6, 256, 256, device=device)
    invalid = torch.zeros(1, 1, 256, 256, device=device)
    invalid[:, :, :64, :64] = 1
    label = torch.zeros(1, 256, 256, dtype=torch.long, device=device)
    label[:, 64:128, :128] = 1
    label[:, 128:192, 64:192] = 2
    label[:, :64, :64] = IGNORE_INDEX
    return x, invalid, label


def test_resnet50_three_class_two_head_forward_backward_and_ignore():
    device = torch.device("cpu")
    x, invalid, label = synthetic_batch(device)
    validate_batch(x, invalid, label)
    model = make_smagnet_resnet50(AUTHOR_SOURCE).to(device).train()
    s1, s2 = split_s1_s2(x)
    fused, sar, gates = model(s1, s2, invalid)
    assert fused.shape == sar.shape == (1, 3, 256, 256)
    assert len(gates) == 5
    weights = torch.tensor([1.0, 1.2, 2.0], device=device)
    loss, parts = two_head_loss(fused, sar, label, weights)
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, 0.5 * (parts["fused_loss"] + parts["sar_loss"]))
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    altered_fused = fused.detach().clone()
    altered_sar = sar.detach().clone()
    altered_fused[:, :, :64, :64] = 1000
    altered_sar[:, :, :64, :64] = -1000
    ignored_loss, _ = two_head_loss(altered_fused, altered_sar, label, weights)
    torch.testing.assert_close(ignored_loss, loss.detach())
    assert all(torch.isfinite(value) for value in parts.values())


def test_adapter_rejects_invalid_shapes_and_labels():
    x = torch.zeros(1, 6, 256, 256)
    invalid = torch.zeros(1, 1, 256, 256)
    label = torch.zeros(1, 256, 256, dtype=torch.long)
    validate_batch(x, invalid, label)
    with pytest.raises(ValueError):
        split_s1_s2(torch.zeros(1, 5, 256, 256))
    with pytest.raises(ValueError, match="exact shape"):
        validate_batch(x, invalid[:, 0], label)
    with pytest.raises(ValueError, match="binary"):
        validate_batch(x, torch.full_like(invalid, 0.5), label)
    invalid_nan = invalid.clone()
    invalid_nan[:, :, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_batch(x, invalid_nan, label)
    with pytest.raises(ValueError, match="exact shape"):
        validate_batch(x, torch.zeros(1, 1, 128, 256), label)
    with pytest.raises(ValueError):
        validate_batch(x, invalid, label.to(torch.int32))
    label[:, 0, 0] = 3
    with pytest.raises(ValueError):
        validate_batch(x, invalid, label)


def test_two_head_loss_rejects_all_ignore_synthetic_label():
    logits = torch.zeros(1, 3, 4, 4)
    all_ignore = torch.full((1, 4, 4), IGNORE_INDEX, dtype=torch.long)
    weights = torch.tensor([1.0, 1.2, 2.0])
    with pytest.raises(ValueError, match="no non-Ignore pixels"):
        two_head_loss(logits, logits, all_ignore, weights)

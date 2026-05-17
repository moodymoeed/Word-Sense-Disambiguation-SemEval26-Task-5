"""Standalone unit tests for the multi-objective loss module.

Run with:
    python src/test_losses.py

No GPU or DeBERTa download is required.  All tests use small CPU tensors.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from custom_losses import (
    HuberLoss,
    MultiObjectiveLoss,
    RankNetLoss,
    UncertaintyAwareMarginLoss,
)

_PASS = "\033[32mPASS\033[0m"
_FAIL = "\033[31mFAIL\033[0m"


def check(name: str, condition: bool) -> None:
    print(f"  [{_PASS if condition else _FAIL}] {name}")
    if not condition:
        raise AssertionError(f"Test failed: {name}")


# ---------------------------------------------------------------------------

def test_huber_loss() -> None:
    print("\n--- HuberLoss ---")
    fn = HuberLoss(delta=1.0)

    preds   = torch.tensor([2.0, 3.0, 4.0], requires_grad=True)
    targets = torch.tensor([1.0, 3.0, 5.0])

    loss = fn(preds, targets)
    check("returns positive value for imperfect predictions", loss.item() > 0)
    check("no NaN", not torch.isnan(loss))

    loss.backward()
    check("gradient flows to predictions", preds.grad is not None)
    check("no NaN in gradients", not torch.isnan(preds.grad).any())

    # Perfect prediction -> loss ≈ 0
    perfect = torch.tensor([1.0, 3.0, 5.0], requires_grad=True)
    check("perfect prediction -> loss < 1e-6", fn(perfect, targets).item() < 1e-6)


# ---------------------------------------------------------------------------

def test_ranknet_loss() -> None:
    print("\n--- RankNetLoss ---")
    fn = RankNetLoss(margin=0.0)

    targets   = torch.tensor([1.0, 3.0, 5.0])
    good_preds = torch.tensor([1.2, 3.1, 4.8], requires_grad=True)
    bad_preds  = torch.tensor([5.0, 3.0, 1.0], requires_grad=True)

    good_loss = fn(good_preds, targets)
    bad_loss  = fn(bad_preds,  targets)

    check("correctly-ordered predictions give lower loss", good_loss.item() < bad_loss.item())
    check("no NaN (good preds)", not torch.isnan(good_loss))
    check("no NaN (bad preds)",  not torch.isnan(bad_loss))

    good_loss.backward()
    check("gradient flows for good preds", good_preds.grad is not None)

    # All-same targets -> no valid pairs -> zero loss with working backward
    same_targets = torch.tensor([3.0, 3.0, 3.0])
    same_preds   = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    zero_loss    = fn(same_preds, same_targets)
    check("no valid pairs -> zero loss", zero_loss.item() == 0.0)
    zero_loss.backward()
    check("backward works with zero valid pairs", same_preds.grad is not None)

    # Exactly correct ordering -> loss below sigmoid midpoint
    perfect_preds   = torch.tensor([1.0, 3.0, 5.0], requires_grad=True)
    perfect_targets = torch.tensor([1.0, 3.0, 5.0])
    check("perfectly ordered predictions -> loss < 1.0",
          fn(perfect_preds, perfect_targets).item() < 1.0)


# ---------------------------------------------------------------------------

def test_uncertainty_loss() -> None:
    print("\n--- UncertaintyAwareMarginLoss ---")
    fn = UncertaintyAwareMarginLoss(squared=False)

    targets = torch.tensor([1.0, 3.0, 5.0])
    stdevs  = torch.tensor([0.5, 0.5, 0.5])

    # Predictions strictly inside the tolerance band -> zero loss
    inside_preds = torch.tensor([1.3, 2.8, 4.7], requires_grad=True)
    inside_loss  = fn(inside_preds, targets, stdevs)
    check("prediction inside stdev band -> zero loss", inside_loss.item() < 1e-6)

    # Predictions outside the tolerance band -> positive loss
    outside_preds = torch.tensor([3.0, 1.0, 3.0], requires_grad=True)
    outside_loss  = fn(outside_preds, targets, stdevs)
    check("prediction outside stdev band -> positive loss", outside_loss.item() > 0)
    check("no NaN", not torch.isnan(outside_loss))

    outside_loss.backward()
    check("gradient flows for outside predictions", outside_preds.grad is not None)

    # Squared variant returns positive
    sq_fn   = UncertaintyAwareMarginLoss(squared=True)
    sq_loss = sq_fn(outside_preds.detach().requires_grad_(True), targets, stdevs)
    check("squared variant returns positive", sq_loss.item() > 0)

    # Negative stdev is clamped to 0 and treated as a point-loss (no tolerance)
    neg_stdevs   = torch.tensor([-0.5, -0.5, -0.5])
    neg_preds    = torch.tensor([1.0, 3.0, 5.0], requires_grad=True)
    neg_stdev_loss = fn(neg_preds, targets, neg_stdevs)
    check("negative stdev is safely clamped (no crash)", not torch.isnan(neg_stdev_loss))


# ---------------------------------------------------------------------------

def test_combined_loss() -> None:
    print("\n--- MultiObjectiveLoss ---")
    fn = MultiObjectiveLoss(
        lambda_huber=1.0,
        lambda_ranknet=0.5,
        lambda_uncertainty=0.5,
    )

    targets = torch.tensor([1.0, 3.0, 5.0])
    stdevs  = torch.tensor([0.5, 0.5, 0.5])
    preds   = torch.tensor([1.2, 3.1, 4.8], requires_grad=True)

    result = fn(predictions=preds, targets=targets, stdevs=stdevs)

    check("returns a dict",                 isinstance(result, dict))
    check("dict has 'loss' key",            "loss"             in result)
    check("dict has 'huber_loss' key",      "huber_loss"       in result)
    check("dict has 'ranknet_loss' key",    "ranknet_loss"     in result)
    check("dict has 'uncertainty_loss' key","uncertainty_loss" in result)
    check("total loss is positive",         result["loss"].item() > 0)
    check("no NaN in total loss",           not torch.isnan(result["loss"]))

    result["loss"].backward()
    check("backward pass works",             preds.grad is not None)
    check("no NaN in gradients",             not torch.isnan(preds.grad).any())

    # Weighted sum check: total ≈ 1.0*huber + 0.5*ranknet + 0.5*uncertainty
    expected = (
        1.0 * result["huber_loss"].item()
        + 0.5 * result["ranknet_loss"].item()
        + 0.5 * result["uncertainty_loss"].item()
    )
    check("total equals weighted sum of components",
          abs(result["loss"].item() - expected) < 1e-5)


# ---------------------------------------------------------------------------

def test_shape_handling() -> None:
    print("\n--- Shape Handling ---")
    fn      = MultiObjectiveLoss()
    targets = torch.tensor([2.0, 4.0])
    stdevs  = torch.tensor([0.3, 0.7])

    for desc, preds in [
        ("[batch]",    torch.tensor([2.1, 3.9], requires_grad=True)),
        ("[batch, 1]", torch.tensor([[2.1], [3.9]], requires_grad=True)),
    ]:
        result = fn(preds, targets, stdevs)
        check(f"{desc}: no NaN in loss",   not torch.isnan(result["loss"]))
        result["loss"].backward()
        check(f"{desc}: gradient exists",  preds.grad is not None)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running multi-objective loss tests (CPU only, no DeBERTa required)...")
    test_huber_loss()
    test_ranknet_loss()
    test_uncertainty_loss()
    test_combined_loss()
    test_shape_handling()
    print("\n\033[32mAll tests passed!\033[0m")

"""Multi-objective loss engineering for the CS445 Group 2 SemEval system.

Project Phase: Step 4 - Mathematical Optimization
--------------------------------------------------
This module provides three complementary loss components and a combined wrapper.

- HuberLoss:                  Robust absolute regression accuracy (outlier-safe MSE).
- RankNetLoss:                Pairwise ranking loss that directly optimises Spearman-style ordering.
- UncertaintyAwareMarginLoss: Respects human annotator disagreement; stdev acts as a zero-loss band.
- MultiObjectiveLoss:         Weighted combination of all three objectives.

Motivation (U-shaped label distribution):
  Simple MSE is blind to rank ordering and treats all errors equally regardless of how
  strongly annotators agreed. The AmbiStory dataset has many ratings near 1 and 5
  (bimodal), so we need a loss that simultaneously targets:
    1. Absolute score accuracy      → Huber
    2. Correct pairwise ordering    → RankNet
    3. Tolerance to disagreement    → Uncertainty-aware margin

Upstream:  SequenceModel (sequence_model.py) returns predictions shape (batch, 1).
Upstream:  DataLoader (data_loader.py) yields 'target_score' and 'target_stdev' tensors.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Huber Loss
# ---------------------------------------------------------------------------

class HuberLoss(nn.Module):
    """Robust regression loss that blends MSE and MAE.

    Huber = absolute score accuracy, robust to outliers at the distribution tails.

    Regime:
        |error| <= delta  →  loss = 0.5 * error²   (smooth quadratic near target)
        |error|  > delta  →  loss = delta * (|error| - 0.5 * delta)   (linear)
    """

    def __init__(self, delta: float = 1.0) -> None:
        super().__init__()
        self._fn = nn.HuberLoss(delta=delta, reduction="mean")

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        return self._fn(predictions.view(-1), targets.view(-1))


# ---------------------------------------------------------------------------
# RankNet Loss
# ---------------------------------------------------------------------------

class RankNetLoss(nn.Module):
    """Pairwise ranking loss — RankNet = ranking / Spearman-style ordering.

    For every valid pair (i, j) where target_i > target_j + margin the model
    should satisfy prediction_i > prediction_j.  Violations are penalised with:

        loss(i,j) = log(1 + exp(-(pred_i - pred_j)))
                  = softplus(-(pred_i - pred_j))          # numerically stable

    This is equivalent to −log σ(pred_i − pred_j), i.e. pushing the sigmoid
    probability of the correct ordering toward 1.

    All B² pairs are evaluated without Python loops via broadcasting, then the
    subset of valid pairs is selected via a boolean mask before averaging.

    If no valid pairs exist in a batch (e.g. all targets identical) the method
    returns a differentiable zero so the backward pass still runs cleanly.
    """

    def __init__(self, margin: float = 0.0) -> None:
        """
        Args:
            margin: Minimum difference in target scores for a pair to count as
                    a valid ranking pair.  Pairs where |target_i − target_j| ≤
                    margin are ignored (they are too close to determine order).
        """
        super().__init__()
        self.margin = margin

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        preds = predictions.view(-1)   # [B]
        tgts = targets.view(-1)        # [B]

        # Outer-product broadcasting — no Python loops over pairs
        pred_i = preds.unsqueeze(1)    # [B, 1]
        pred_j = preds.unsqueeze(0)    # [1, B]
        tgt_i  = tgts.unsqueeze(1)     # [B, 1]
        tgt_j  = tgts.unsqueeze(0)     # [1, B]

        # i should rank above j (strict inequality avoids symmetric double-counting)
        valid_mask = (tgt_i - tgt_j) > self.margin  # [B, B]

        if not valid_mask.any():
            # Differentiable zero — preserves grad_fn so .backward() still works
            return (preds * 0.0).sum()

        score_diff = pred_i - pred_j          # [B, B]; positive = correct order
        pair_loss  = F.softplus(-score_diff)  # [B, B]; 0 → well-ordered, large → violation

        return pair_loss[valid_mask].mean()


# ---------------------------------------------------------------------------
# Uncertainty-Aware Margin Loss
# ---------------------------------------------------------------------------

class UncertaintyAwareMarginLoss(nn.Module):
    """Margin loss — uncertainty-aware margin = respects human disagreement.

    Human annotators naturally disagree; that disagreement is captured by the
    stdev field.  A prediction inside [target − stdev, target + stdev] is
    treated as acceptable (loss = 0).  Only the excess beyond this band incurs
    a penalty, preventing the model from over-fitting to noise.

    Loss formula:
        residual = |prediction − target|
        excess   = relu(residual − stdev)       # zero inside the tolerance band
        loss     = mean(excess)                  # linear (default, as per report)
               or  mean(excess²)                 # if squared=True
    """

    def __init__(self, squared: bool = False) -> None:
        """
        Args:
            squared: If True, penalise excess² rather than excess.  The linear
                     mode (default) matches the project report description.
        """
        super().__init__()
        self.squared = squared

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        stdevs: torch.Tensor,
    ) -> torch.Tensor:
        preds  = predictions.view(-1)              # [B]
        tgts   = targets.view(-1)                  # [B]
        sigmas = stdevs.view(-1).clamp(min=0.0)   # [B]; guard against negative stdev

        residual = (preds - tgts).abs()            # [B]
        excess   = F.relu(residual - sigmas)       # [B]; zero inside tolerance band

        if self.squared:
            excess = excess ** 2

        return excess.mean()


# ---------------------------------------------------------------------------
# Combined Multi-Objective Loss
# ---------------------------------------------------------------------------

class MultiObjectiveLoss(nn.Module):
    """Weighted combination of Huber, RankNet, and uncertainty-aware margin losses.

    total_loss = lambda_huber       * huber_loss
               + lambda_ranknet     * ranknet_loss
               + lambda_uncertainty * uncertainty_loss

    Lambda weights let callers balance the three objectives during training.
    The forward method returns a dictionary so every component can be logged
    to TensorBoard or a CSV without extra bookkeeping in the training loop.

    Expected usage::

        criterion = MultiObjectiveLoss(
            lambda_huber=1.0,
            lambda_ranknet=0.5,
            lambda_uncertainty=0.5,
        )
        loss_dict = criterion(
            predictions=predictions,   # (batch,) or (batch, 1)
            targets=batch["target_score"],
            stdevs=batch["target_stdev"],
        )
        loss_dict["loss"].backward()
    """

    def __init__(
        self,
        lambda_huber: float = 1.0,
        lambda_ranknet: float = 0.5,
        lambda_uncertainty: float = 0.5,
        huber_delta: float = 1.0,
        ranknet_margin: float = 0.0,
        uncertainty_squared: bool = False,
    ) -> None:
        """
        Args:
            lambda_huber:         Weight for the Huber regression loss.
            lambda_ranknet:       Weight for the RankNet pairwise ranking loss.
            lambda_uncertainty:   Weight for the uncertainty-aware margin loss.
            huber_delta:          Transition point for the Huber quadratic/linear boundary.
            ranknet_margin:       Minimum target-score gap to consider a pair valid.
            uncertainty_squared:  If True, square the margin excess penalty.
        """
        super().__init__()
        self.lambda_huber       = lambda_huber
        self.lambda_ranknet     = lambda_ranknet
        self.lambda_uncertainty = lambda_uncertainty

        self.huber       = HuberLoss(delta=huber_delta)
        self.ranknet     = RankNetLoss(margin=ranknet_margin)
        self.uncertainty = UncertaintyAwareMarginLoss(squared=uncertainty_squared)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        stdevs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute the combined multi-objective loss.

        Args:
            predictions: Model plausibility predictions, shape [batch] or [batch, 1].
            targets:     Human average plausibility scores, shape [batch].
            stdevs:      Human annotation standard deviations, shape [batch].

        Returns:
            dict with keys:
                "loss"             — total weighted loss; call .backward() on this
                "huber_loss"       — Huber component (for logging)
                "ranknet_loss"     — RankNet component (for logging)
                "uncertainty_loss" — uncertainty-aware margin component (for logging)
        """
        huber_val       = self.huber(predictions, targets)
        ranknet_val     = self.ranknet(predictions, targets)
        uncertainty_val = self.uncertainty(predictions, targets, stdevs)

        total = (
            self.lambda_huber       * huber_val
            + self.lambda_ranknet   * ranknet_val
            + self.lambda_uncertainty * uncertainty_val
        )

        return {
            "loss":             total,
            "huber_loss":       huber_val,
            "ranknet_loss":     ranknet_val,
            "uncertainty_loss": uncertainty_val,
        }

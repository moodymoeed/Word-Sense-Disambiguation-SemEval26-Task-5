"""Main training loop for the CS445 Group 2 SemEval system.

This script wires together all four pipeline stages:
  Stage 1 — Data loading        (data_loader.py      → Raid)
  Stage 2 — Feature extraction  (feature_extraction.py → Moeed)
  Stage 3 — Sequence modeling   (sequence_model.py   → Areeb)
  Stage 4 — Multi-objective loss (custom_losses.py   → Ramzy)

Usage:
    # From the repo root
    python src/train.py

    # With custom hyperparameters
    python src/train.py --epochs 5 --batch_size 4 --lr 1e-5 \
        --lambda_huber 1.0 --lambda_ranknet 0.5 --lambda_uncertainty 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import build_dataloader, build_tokenizer
from feature_extraction import HierarchicalDebertaEncoder
from sequence_model import SequenceModel
from custom_losses import MultiObjectiveLoss


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CS445 Group 2 SemEval model.")
    p.add_argument("--data_dir",           type=str,   default="data")
    p.add_argument("--model_name",         type=str,   default="microsoft/deberta-large")
    p.add_argument("--epochs",             type=int,   default=10)
    p.add_argument("--batch_size",         type=int,   default=8)
    p.add_argument("--lr",                 type=float, default=2e-5)
    p.add_argument("--lambda_huber",       type=float, default=1.0)
    p.add_argument("--lambda_ranknet",     type=float, default=0.5)
    p.add_argument("--lambda_uncertainty", type=float, default=0.5)
    p.add_argument("--huber_delta",        type=float, default=1.0)
    p.add_argument("--ranknet_margin",     type=float, default=0.0)
    p.add_argument("--grad_clip",          type=float, default=1.0)
    p.add_argument("--log_every",          type=int,   default=50,
                   help="Print a progress line every N batches.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def _evaluate(
    encoder:   HierarchicalDebertaEncoder,
    model:     SequenceModel,
    loader,
    criterion: MultiObjectiveLoss,
    device:    torch.device,
) -> dict:
    encoder.eval()
    model.eval()

    totals = {"loss": 0.0, "huber_loss": 0.0, "ranknet_loss": 0.0, "uncertainty_loss": 0.0}
    n = 0

    for batch in loader:
        targets = batch["target_score"].to(device)
        stdevs  = batch["target_stdev"].to(device)
        batch_d = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in batch.items()}

        embeddings  = encoder(batch_d)
        predictions = model(embeddings)
        loss_dict   = criterion(predictions=predictions, targets=targets, stdevs=stdevs)

        for key in totals:
            totals[key] += loss_dict[key].item()
        n += 1

    n = max(n, 1)
    return {k: v / n for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 1. Data ──────────────────────────────────────────────────────────────
    print("\n[1/4] Building tokeniser and data loaders...")
    tokenizer    = build_tokenizer(args.model_name)
    data_root    = Path(args.data_dir)

    train_loader = build_dataloader(
        data_path=data_root / "train.json",
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        shuffle=True,
    )
    dev_loader = build_dataloader(
        data_path=data_root / "dev.json",
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        shuffle=False,
    )
    print(f"  Train batches: {len(train_loader)} | Dev batches: {len(dev_loader)}")

    # ── 2. Model ─────────────────────────────────────────────────────────────
    print("\n[2/4] Initialising encoder and sequence model...")
    encoder = HierarchicalDebertaEncoder(model_name=args.model_name).to(device)
    model   = SequenceModel().to(device)

    # Only LoRA adapter parameters in the encoder carry gradients;
    # all SequenceModel parameters are always trainable.
    trainable = (
        list(filter(lambda p: p.requires_grad, encoder.parameters()))
        + list(model.parameters())
    )
    optimizer = optim.AdamW(trainable, lr=args.lr, weight_decay=1e-2)

    # ── 3. Loss ──────────────────────────────────────────────────────────────
    print("\n[3/4] Configuring multi-objective loss...")
    criterion = MultiObjectiveLoss(
        lambda_huber=args.lambda_huber,
        lambda_ranknet=args.lambda_ranknet,
        lambda_uncertainty=args.lambda_uncertainty,
        huber_delta=args.huber_delta,
        ranknet_margin=args.ranknet_margin,
    )
    print(
        f"  λ_huber={args.lambda_huber}  "
        f"λ_ranknet={args.lambda_ranknet}  "
        f"λ_uncertainty={args.lambda_uncertainty}"
    )

    # ── 4. Training loop ─────────────────────────────────────────────────────
    print(f"\n[4/4] Training for {args.epochs} epoch(s)...")
    for epoch in range(1, args.epochs + 1):
        encoder.train()
        model.train()

        totals = {"loss": 0.0, "huber_loss": 0.0, "ranknet_loss": 0.0, "uncertainty_loss": 0.0}
        n_batches = 0

        for step, batch in enumerate(train_loader, 1):
            targets = batch["target_score"].to(device)
            stdevs  = batch["target_stdev"].to(device)
            batch_d = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                       for k, v in batch.items()}

            optimizer.zero_grad()

            embeddings  = encoder(batch_d)
            predictions = model(embeddings)

            # ── Loss computation (Ramzy's multi-objective criterion) ──────────
            loss_dict = criterion(
                predictions=predictions,
                targets=targets,
                stdevs=stdevs,
            )

            loss = loss_dict["loss"]
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=args.grad_clip)

            optimizer.step()

            for key in totals:
                totals[key] += loss_dict[key].item()
            n_batches += 1

            if step % args.log_every == 0:
                avg = totals["loss"] / n_batches
                print(
                    f"  Epoch {epoch} | step {step:>4}/{len(train_loader)} "
                    f"| avg_loss={avg:.4f}"
                )

        n = max(n_batches, 1)
        dev_metrics = _evaluate(encoder, model, dev_loader, criterion, device)

        print(
            f"\nEpoch {epoch}/{args.epochs}"
            f"\n  train  loss={totals['loss']/n:.4f}"
            f"  huber={totals['huber_loss']/n:.4f}"
            f"  ranknet={totals['ranknet_loss']/n:.4f}"
            f"  uncertainty={totals['uncertainty_loss']/n:.4f}"
            f"\n  dev    loss={dev_metrics['loss']:.4f}"
            f"  huber={dev_metrics['huber_loss']:.4f}"
            f"  ranknet={dev_metrics['ranknet_loss']:.4f}"
            f"  uncertainty={dev_metrics['uncertainty_loss']:.4f}\n"
        )


if __name__ == "__main__":
    train(_parse_args())

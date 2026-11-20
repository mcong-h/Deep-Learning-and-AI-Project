"""Train the first pairwise garment-compatibility model.

Run from the code directory:

    python train.py

The default experiment uses toy structured garments and weak/synthetic pair labels.
It is meant to validate the training pipeline before real image encoders and real
care-label data are connected.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random

import torch
from torch import nn
from torch.utils.data import DataLoader

from compatibility_model import CompatibilityMLP
from dataset import GarmentPairDataset, build_pairs, build_toy_garments, split_garments
from encoder import StructuredGarmentEncoder


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for z_a, z_b, labels in loader:
            z_a = z_a.to(device)
            z_b = z_b.to(device)
            labels = labels.to(device)

            logits = model(z_a, z_b)
            loss = criterion(logits, labels)

            probs = torch.sigmoid(logits)
            predictions = (probs >= 0.5).float()

            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-garments", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output",
        type=str,
        default="../results/compatibility_model.pt",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    # Step 1: create garment records.
    garments = build_toy_garments(args.num_garments, seed=args.seed)

    # Split by garment before creating pairs to avoid leakage.
    train_garments, val_garments, test_garments = split_garments(
        garments,
        seed=args.seed,
    )

    # Step 2: encode each garment into z_i.
    encoder = StructuredGarmentEncoder()

    train_pairs = build_pairs(train_garments, max_pairs=1500, seed=args.seed)
    val_pairs = build_pairs(val_garments, max_pairs=400, seed=args.seed)
    test_pairs = build_pairs(test_garments, max_pairs=400, seed=args.seed)

    train_dataset = GarmentPairDataset(train_pairs, encoder)
    val_dataset = GarmentPairDataset(val_pairs, encoder)
    test_dataset = GarmentPairDataset(test_pairs, encoder)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    print(
        f"Pairs: train={len(train_dataset)}, "
        f"val={len(val_dataset)}, test={len(test_dataset)}"
    )

    # Step 3: create our trainable compatibility model.
    model = CompatibilityMLP(
        garment_feature_dim=encoder.output_dim,
        hidden_dim=128,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    best_val_loss = float("inf")
    best_state = None

    # Step 4: training loop.
    for epoch in range(1, args.epochs + 1):
        model.train()

        running_loss = 0.0
        n_examples = 0

        for z_a, z_b, labels in train_loader:
            z_a = z_a.to(device)
            z_b = z_b.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(z_a, z_b)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            n_examples += labels.size(0)

        train_loss = running_loss / max(n_examples, 1)
        val_metrics = evaluate(model, val_loader, criterion, device)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        print(
            f"Epoch {epoch:02d} | "
            f"train loss={train_loss:.4f} | "
            f"val loss={val_metrics['loss']:.4f} | "
            f"val acc={val_metrics['accuracy']:.3f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, criterion, device)

    print("\nFinal test results")
    print(f"test loss={test_metrics['loss']:.4f}")
    print(f"test accuracy={test_metrics['accuracy']:.3f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "garment_feature_dim": encoder.output_dim,
            "test_metrics": test_metrics,
            "note": (
                "Toy structured-data model. Replace synthetic labels/features "
                "with real garment/care-label perception in later stages."
            ),
        },
        output_path,
    )

    print(f"\nSaved model to: {output_path}")


if __name__ == "__main__":
    main()

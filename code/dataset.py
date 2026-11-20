"""Dataset utilities for pairwise garment compatibility training.

The first version intentionally supports a synthetic/structured mode so the full
training pipeline can be tested before connecting real images and care-label OCR.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import random
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset


MATERIALS = [
    "cotton",
    "cotton_linen",
    "cotton_polyester",
    "silk",
    "rayon",
    "polyester",
    "wool_polyester",
    "wool",
]

GARMENT_TYPES = ["tshirt", "shirt", "hoodie", "sweater", "pants", "blouse"]
COLORS = ["light", "dark", "white", "red"]

# A small rule table used only to generate weak/synthetic supervision.
# The final project should replace/refine these labels using the selected
# literature-derived compatibility rules and real garment metadata.
INCOMPATIBLE_MATERIAL_PAIRS = {
    frozenset(("wool", "cotton")),
    frozenset(("wool", "cotton_linen")),
    frozenset(("wool", "cotton_polyester")),
    frozenset(("wool", "polyester")),
    frozenset(("silk", "cotton")),
    frozenset(("silk", "polyester")),
    frozenset(("rayon", "wool")),
}


@dataclass(frozen=True)
class GarmentRecord:
    garment_id: str
    material: str
    garment_type: str
    color: str
    max_wash_temp_c: int
    delicate_cycle: bool


def is_unsafe_pair(a: GarmentRecord, b: GarmentRecord) -> int:
    """Generate a weak/synthetic safe(0)/unsafe(1) label for a garment pair."""
    material_pair = frozenset((a.material, b.material))

    material_conflict = material_pair in INCOMPATIBLE_MATERIAL_PAIRS
    temp_conflict = abs(a.max_wash_temp_c - b.max_wash_temp_c) >= 20
    delicate_conflict = (a.delicate_cycle != b.delicate_cycle) and (
        a.material in {"silk", "wool", "rayon"}
        or b.material in {"silk", "wool", "rayon"}
    )

    return int(material_conflict or temp_conflict or delicate_conflict)


def build_toy_garments(n: int = 80, seed: int = 7) -> List[GarmentRecord]:
    """Create deterministic toy garments for testing the training pipeline."""
    rng = random.Random(seed)
    garments: List[GarmentRecord] = []

    for i in range(n):
        material = rng.choice(MATERIALS)
        garment_type = rng.choice(GARMENT_TYPES)
        color = rng.choice(COLORS)

        if material in {"silk", "wool", "rayon"}:
            temp = rng.choice([20, 30])
            delicate = True
        else:
            temp = rng.choice([30, 40, 60])
            delicate = rng.random() < 0.15

        garments.append(
            GarmentRecord(
                garment_id=f"garment_{i:03d}",
                material=material,
                garment_type=garment_type,
                color=color,
                max_wash_temp_c=temp,
                delicate_cycle=delicate,
            )
        )

    return garments


def split_garments(
    garments: Sequence[GarmentRecord],
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    seed: int = 7,
) -> Tuple[List[GarmentRecord], List[GarmentRecord], List[GarmentRecord]]:
    """Split by garment, not by pair, to avoid pair-level data leakage."""
    items = list(garments)
    rng = random.Random(seed)
    rng.shuffle(items)

    n = len(items)
    n_train = int(n * train_fraction)
    n_val = int(n * val_fraction)

    train = items[:n_train]
    val = items[n_train : n_train + n_val]
    test = items[n_train + n_val :]
    return train, val, test


def build_pairs(
    garments: Sequence[GarmentRecord],
    max_pairs: int | None = None,
    seed: int = 7,
) -> List[Tuple[GarmentRecord, GarmentRecord, int]]:
    """Generate garment pairs and weak/synthetic compatibility labels."""
    pairs = [(a, b, is_unsafe_pair(a, b)) for a, b in combinations(garments, 2)]

    if max_pairs is not None and len(pairs) > max_pairs:
        rng = random.Random(seed)
        rng.shuffle(pairs)
        pairs = pairs[:max_pairs]

    return pairs


class GarmentPairDataset(Dataset):
    """Pair dataset using a callable garment encoder."""

    def __init__(
        self,
        pairs: Iterable[Tuple[GarmentRecord, GarmentRecord, int]],
        encoder,
    ) -> None:
        self.pairs = list(pairs)
        self.encoder = encoder

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        garment_a, garment_b, label = self.pairs[index]

        z_a = self.encoder(garment_a)
        z_b = self.encoder(garment_b)

        return (
            torch.as_tensor(z_a, dtype=torch.float32),
            torch.as_tensor(z_b, dtype=torch.float32),
            torch.tensor(float(label), dtype=torch.float32),
        )

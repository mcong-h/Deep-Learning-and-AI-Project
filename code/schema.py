"""Shared data structures for the laundry-planning pipeline.

This module defines the interface between perception, planning, and evaluation.
The perception system should output GarmentProfile objects. Planners should use
those structured profiles instead of working directly with raw images.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


def _validate_distribution(distribution: Dict[str, float]) -> None:
    """Validate that confidence values form a non-empty probability-like map."""
    if not distribution:
        raise ValueError("Prediction distribution cannot be empty.")

    for label, score in distribution.items():
        if not 0.0 <= score <= 1.0:
            raise ValueError(
                f"Confidence for {label!r} must be between 0 and 1, got {score}."
            )


@dataclass
class AttributePrediction:
    """Prediction for one garment attribute with uncertainty information."""

    probabilities: Dict[str, float]

    def __post_init__(self) -> None:
        _validate_distribution(self.probabilities)

    @property
    def top_label(self) -> str:
        """Return the highest-confidence label."""
        return max(self.probabilities, key=self.probabilities.get)

    @property
    def top_confidence(self) -> float:
        """Return the confidence of the highest-probability label."""
        return self.probabilities[self.top_label]

    def plausible_labels(self, min_probability: float = 0.20) -> List[str]:
        """Return labels that remain plausible above a chosen threshold."""
        return [
            label
            for label, probability in self.probabilities.items()
            if probability >= min_probability
        ]


@dataclass
class CareConstraints:
    """Structured washing constraints extracted from a garment care label."""

    max_wash_temp_c: Optional[int] = None
    hand_wash_only: bool = False
    delicate_cycle: bool = False
    bleach_allowed: Optional[bool] = None
    tumble_dry_allowed: Optional[bool] = None
    dry_clean_only: bool = False
    notes: List[str] = field(default_factory=list)


@dataclass
class GroundTruth:
    """Optional ground-truth attributes used only for experiments/evaluation."""

    garment_type: Optional[str] = None
    color: Optional[str] = None
    material: Optional[str] = None
    care_constraints: Optional[CareConstraints] = None


@dataclass
class GarmentProfile:
    """Structured representation passed from perception to the planners."""

    garment_id: str
    garment_type: AttributePrediction
    color: AttributePrediction
    material: AttributePrediction
    care_constraints: CareConstraints
    care_confidence: Optional[float] = None

    garment_image_path: Optional[str] = None
    care_label_image_path: Optional[str] = None

    ground_truth: Optional[GroundTruth] = None

    def __post_init__(self) -> None:
        if self.care_confidence is not None and not 0.0 <= self.care_confidence <= 1.0:
            raise ValueError("care_confidence must be between 0 and 1.")

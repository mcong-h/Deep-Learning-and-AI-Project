"""Garment encoders.

The default StructuredGarmentEncoder makes the training pipeline runnable now.
CLIPImageEncoder is an optional future path for real garment images.
"""

from __future__ import annotations

from typing import Dict, List

import torch

from dataset import COLORS, GARMENT_TYPES, MATERIALS, GarmentRecord


class StructuredGarmentEncoder:
    """Convert structured garment metadata into a numeric feature vector.

    This is intentionally simple and deterministic. It lets us test the pairwise
    compatibility-learning pipeline before real image/care-label perception is
    connected.
    """

    def __init__(self) -> None:
        self.material_index: Dict[str, int] = {
            name: i for i, name in enumerate(MATERIALS)
        }
        self.type_index: Dict[str, int] = {
            name: i for i, name in enumerate(GARMENT_TYPES)
        }
        self.color_index: Dict[str, int] = {
            name: i for i, name in enumerate(COLORS)
        }

        self.output_dim = (
            len(MATERIALS)
            + len(GARMENT_TYPES)
            + len(COLORS)
            + 2
        )

    def __call__(self, garment: GarmentRecord) -> torch.Tensor:
        features: List[float] = [0.0] * self.output_dim

        offset = 0
        features[offset + self.material_index[garment.material]] = 1.0
        offset += len(MATERIALS)

        features[offset + self.type_index[garment.garment_type]] = 1.0
        offset += len(GARMENT_TYPES)

        features[offset + self.color_index[garment.color]] = 1.0
        offset += len(COLORS)

        # Normalize temperature to roughly 0-1 for this toy setup.
        features[offset] = garment.max_wash_temp_c / 60.0
        features[offset + 1] = float(garment.delicate_cycle)

        return torch.tensor(features, dtype=torch.float32)


class CLIPImageEncoder:
    """Optional pretrained CLIP image encoder for future real-image experiments.

    Requires:
        pip install transformers pillow

    Example:
        encoder = CLIPImageEncoder()
        z = encoder.encode_image("data/images/shirt.jpg")

    This class is not used by the default toy training script.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise ImportError(
                "CLIPImageEncoder requires 'transformers' and 'pillow'."
            ) from exc

        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.model.eval()

        # Keep the pretrained encoder frozen in the first project version.
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.output_dim = self.model.config.projection_dim

    @torch.no_grad()
    def encode_image(self, image_path: str) -> torch.Tensor:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        features = self.model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu()

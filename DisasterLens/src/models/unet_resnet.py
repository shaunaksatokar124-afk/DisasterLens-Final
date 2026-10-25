"""
DisasterLens AI -- U-Net + ResNet Model
Wraps segmentation_models_pytorch to provide the damage segmentation network.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def build_model(
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
    in_channels: int = 6,
    num_classes: int = 5,
) -> nn.Module:
    """Build a U-Net segmentation model with a pretrained ResNet encoder.

    Args:
        encoder_name:    Encoder backbone (e.g. "resnet34", "resnet50", "efficientnet-b3").
        encoder_weights: Pretrained weight source ("imagenet" or None).
                         When in_channels != 3, SMP averages the pretrained first-layer
                         weights -- this is safe and well-established.
        in_channels:     Input channels: 6 = 3-ch pre-disaster + 3-ch post-disaster.
        num_classes:     Output classes: 5 = background + 4 damage levels.

    Returns:
        U-Net nn.Module with raw logit outputs.

    Raises:
        ImportError: If segmentation_models_pytorch is not installed.
    """
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise ImportError(
            "segmentation_models_pytorch is required. "
            "Install with: pip install segmentation-models-pytorch"
        ) from exc

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,                   # Raw logits; softmax applied during inference / loss
        decoder_attention_type="scse",     # Squeeze-and-Excitation attention in decoder
    )
    return model


class DisasterLensModel(nn.Module):
    """Named wrapper for checkpoint serialisation and UI metadata."""

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 6,
        num_classes: int = 5,
    ) -> None:
        super().__init__()
        self.encoder_name = encoder_name
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.network = build_model(
            encoder_name, encoder_weights, in_channels, num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W) normalised float32 tensor.
        Returns:
            (B, num_classes, H, W) logit tensor.
        """
        return self.network(x)

    def get_model_info(self) -> dict:
        """Return metadata dict for UI display."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "architecture": f"U-Net + {self.encoder_name}",
            "encoder": self.encoder_name,
            "in_channels": self.in_channels,
            "num_classes": self.num_classes,
            "total_parameters": f"{total:,}",
            "trainable_parameters": f"{trainable:,}",
        }

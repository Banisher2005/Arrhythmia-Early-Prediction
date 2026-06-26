"""
Multi-Scale Residual Feature Extraction Block

Runs three parallel residual branches with different kernel sizes
and fuses them using the Adaptive Gated Feature Fusion module.

This module forms the core feature extractor of AMSRAN-GF.
"""

import torch
import torch.nn as nn

from src.models.adaptive_gate import AdaptiveGate
from src.models.residual_block import ResidualBlock
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MultiScaleBlock(nn.Module):
    """
    Multi-scale residual feature extraction.

    Input
        (B, C, L)

    Output
        (B, C_out, L)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()

        self.branch3 = ResidualBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            dropout=dropout,
        )

        self.branch5 = ResidualBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=5,
            dropout=dropout,
        )

        self.branch7 = ResidualBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=7,
            dropout=dropout,
        )

        self.gate = AdaptiveGate(
            channels=out_channels,
            num_branches=3,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape:
                (B, C, L)

        Returns
        -------
        fused_features
            Shape:
                (B, C_out, L)

        gate_weights
            Shape:
                (B, 3)
        """

        feature3 = self.branch3(x)

        feature5 = self.branch5(x)

        feature7 = self.branch7(x)

        fused, gate_weights = self.gate(
            [
                feature3,
                feature5,
                feature7,
            ]
        )

        return fused, gate_weights


def main() -> None:
    """
    Example usage.
    """

    model = MultiScaleBlock(
        in_channels=1,
        out_channels=64,
    )

    x = torch.randn(
        8,
        1,
        180,
    )

    fused, weights = model(x)

    logger.info(
        "Input Shape : %s",
        tuple(x.shape),
    )

    logger.info(
        "Output Shape : %s",
        tuple(fused.shape),
    )

    logger.info(
        "Gate Shape : %s",
        tuple(weights.shape),
    )

    logger.info(
        "First Sample Gate : %s",
        weights[0].detach().cpu().numpy(),
    )


if __name__ == "__main__":
    main()
"""
Residual CNN Backbone

Stacks multiple Multi-Scale Residual Blocks to extract rich ECG
representations before sequence modeling.

Output:
    (Batch, Sequence Length, Feature Dimension)

This backbone is used by AMSRAN-GF.
"""

import torch
import torch.nn as nn

from src.models.multiscale_block import MultiScaleBlock
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ResidualCNN(nn.Module):
    """
    Multi-Scale Residual CNN Backbone.
    """

    def __init__(
        self,
        input_channels: int = 1,
        feature_dim: int = 64,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()

        self.block1 = MultiScaleBlock(
            in_channels=input_channels,
            out_channels=32,
            dropout=dropout,
        )

        self.pool1 = nn.MaxPool1d(
            kernel_size=2,
            stride=2,
        )

        self.block2 = MultiScaleBlock(
            in_channels=32,
            out_channels=64,
            dropout=dropout,
        )

        self.pool2 = nn.MaxPool1d(
            kernel_size=2,
            stride=2,
        )

        self.block3 = MultiScaleBlock(
            in_channels=64,
            out_channels=feature_dim,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        list[torch.Tensor],
    ]:
        """
        Parameters
        ----------
        x
            Shape:
                (B,1,180)

        Returns
        -------
        features
            Shape:
                (B,L,C)

        gate_weights
            Gate outputs from all blocks.
        """

        gate_outputs = []

        x, gate = self.block1(x)
        gate_outputs.append(gate)

        x = self.pool1(x)

        x, gate = self.block2(x)
        gate_outputs.append(gate)

        x = self.pool2(x)

        x, gate = self.block3(x)
        gate_outputs.append(gate)

        x = x.permute(
            0,
            2,
            1,
        )

        return x, gate_outputs


def main() -> None:
    """
    Example usage.
    """

    model = ResidualCNN()

    x = torch.randn(
        8,
        1,
        180,
    )

    features, gates = model(x)

    logger.info(
        "CNN Output Shape : %s",
        tuple(features.shape),
    )

    for i, gate in enumerate(
        gates,
        start=1,
    ):
        logger.info(
            "Gate %d Shape : %s",
            i,
            tuple(gate.shape),
        )

        logger.info(
            "Gate %d First Sample : %s",
            i,
            gate[0].detach().cpu().numpy(),
        )


if __name__ == "__main__":
    main()
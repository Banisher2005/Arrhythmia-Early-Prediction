"""
Residual CNN Backbone

Uses stacked Multi-Scale Residual Blocks with learnable downsampling.
"""

import torch
import torch.nn as nn

from src.models.multiscale_block import MultiScaleBlock
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DownsampleBlock(nn.Module):
    """
    Learnable downsampling block.
    """

    def __init__(
        self,
        channels: int,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )

        nn.init.kaiming_normal_(
            self.block[0].weight,
            mode="fan_out",
            nonlinearity="relu",
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.block(x)


class ResidualCNN(nn.Module):
    """
    Multi-Scale CNN Backbone.
    """

    def __init__(
        self,
        input_channels: int = 1,
        feature_dim: int = 64,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()

        self.block1 = MultiScaleBlock(
            input_channels,
            32,
            dropout,
        )

        self.down1 = DownsampleBlock(
            32,
        )

        self.block2 = MultiScaleBlock(
            32,
            64,
            dropout,
        )

        self.down2 = DownsampleBlock(
            64,
        )

        self.block3 = MultiScaleBlock(
            64,
            feature_dim,
            dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        list[torch.Tensor],
    ]:

        gates = []

        x, gate = self.block1(x)
        gates.append(gate)

        x = self.down1(x)

        x, gate = self.block2(x)
        gates.append(gate)

        x = self.down2(x)

        x, gate = self.block3(x)
        gates.append(gate)

        x = x.permute(
            0,
            2,
            1,
        )

        return x, gates


def main() -> None:

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
        "First Block Channel0 Weights : %s",
        gates[0][0, :, 0]
        .detach()
        .cpu()
        .numpy(),
    )


if __name__ == "__main__":
    main()    
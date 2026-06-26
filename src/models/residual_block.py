"""
Residual Block

Reusable 1D residual block for ECG feature extraction.
"""

from typing import Optional

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Standard 1D Residual Block.

    Conv1D
        ↓
    BatchNorm
        ↓
    ReLU
        ↓
    Conv1D
        ↓
    BatchNorm
        ↓
    Skip Connection
        ↓
    ReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )

        self.bn1 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )

        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut: Optional[nn.Module] = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        He initialization.
        """

        for module in self.modules():

            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        """

        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity

        out = self.relu(out)

        return out


def main() -> None:
    """
    Example usage.
    """

    from src.utils.logger import get_logger

    logger = get_logger(__name__)

    model = ResidualBlock(
        in_channels=1,
        out_channels=32,
        kernel_size=5,
    )

    x = torch.randn(
        16,
        1,
        180,
    )

    y = model(x)

    logger.info(
        "Input Shape : %s",
        tuple(x.shape),
    )

    logger.info(
        "Output Shape: %s",
        tuple(y.shape),
    )

if __name__ == "__main__":
    main()
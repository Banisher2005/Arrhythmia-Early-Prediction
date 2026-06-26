"""
Adaptive Gated Feature Fusion

Learns the importance of each feature extraction branch for every ECG
heartbeat.

This module is the primary novelty of the AMSRAN-GF architecture.
"""

import torch
import torch.nn as nn

from src.utils.logger import get_logger

logger = get_logger(__name__)


class AdaptiveGate(nn.Module):
    """
    Adaptive Gated Feature Fusion.

    Given multiple feature maps extracted using different receptive
    fields, the gate learns a set of attention weights indicating
    the importance of each branch.

    Input:
        List[Tensor]
            [(B,C,L), (B,C,L), ...]

    Output:
        Tensor
            (B,C,L)
    """

    def __init__(
        self,
        channels: int,
        num_branches: int = 3,
    ) -> None:
        super().__init__()

        self.channels = channels
        self.num_branches = num_branches

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.gate = nn.Sequential(
            nn.Linear(
                channels * num_branches,
                channels,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                channels,
                num_branches,
            ),
        )

        self.softmax = nn.Softmax(dim=1)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Xavier initialization.
        """

        for module in self.modules():

            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        features: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        features
            List of feature maps.

        Returns
        -------
        fused_features
            Weighted feature map.

        gate_weights
            Branch weights.
            Shape:
                (batch_size, num_branches)
        """

        if len(features) != self.num_branches:
            raise ValueError(
                f"Expected {self.num_branches} branches, "
                f"received {len(features)}."
            )

        pooled = [
            self.global_pool(feature).squeeze(-1)
            for feature in features
        ]

        pooled = torch.cat(
            pooled,
            dim=1,
        )

        gate_weights = self.gate(pooled)

        gate_weights = self.softmax(gate_weights)

        fused = 0

        for i, feature in enumerate(features):

            weight = gate_weights[:, i]

            weight = weight.unsqueeze(1).unsqueeze(2)

            fused = fused + weight * feature

        return fused, gate_weights


def main() -> None:
    """
    Example usage.
    """

    gate = AdaptiveGate(
        channels=64,
        num_branches=3,
    )

    branch1 = torch.randn(
        8,
        64,
        180,
    )

    branch2 = torch.randn(
        8,
        64,
        180,
    )

    branch3 = torch.randn(
        8,
        64,
        180,
    )

    fused, weights = gate(
        [
            branch1,
            branch2,
            branch3,
        ]
    )

    logger.info(
        "Fused Shape : %s",
        tuple(fused.shape),
    )

    logger.info(
        "Gate Shape : %s",
        tuple(weights.shape),
    )

    logger.info(
        "First Sample Weights : %s",
        weights[0].detach().cpu().numpy(),
    )


if __name__ == "__main__":
    main()
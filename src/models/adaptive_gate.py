"""
Channel-wise Adaptive Gated Feature Fusion

Novelty module of AMSRAN-GF.

Learns, for every feature channel, how much importance to assign
to each multi-scale residual branch.
"""

import torch
import torch.nn as nn

from src.utils.logger import get_logger

logger = get_logger(__name__)


class AdaptiveGate(nn.Module):
    """
    Channel-wise Adaptive Gated Feature Fusion.

    Input
    -----
    List[Tensor]
        3 tensors of shape (B, C, L)

    Output
    ------
    fused_features
        (B, C, L)

    gate_weights
        (B, 3, C)
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        num_branches: int = 3,
    ) -> None:
        super().__init__()

        self.channels = channels
        self.num_branches = num_branches

        self.pool = nn.AdaptiveAvgPool1d(1)

        hidden = max(channels // reduction, 4)

        self.fc = nn.Sequential(
            nn.Linear(
                channels * num_branches,
                hidden,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                hidden,
                channels * num_branches,
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

                nn.init.xavier_uniform_(
                    module.weight
                )

                if module.bias is not None:
                    nn.init.zeros_(
                        module.bias
                    )

    def forward(
        self,
        features: list[torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Parameters
        ----------
        features
            List of feature maps.

        Returns
        -------
        fused_features
            (B,C,L)

        gate_weights
            (B,3,C)
        """

        if len(features) != self.num_branches:
            raise ValueError(
                f"Expected {self.num_branches} branches."
            )

        pooled = [
            self.pool(f).squeeze(-1)
            for f in features
        ]

        pooled = torch.cat(
            pooled,
            dim=1,
        )

        weights = self.fc(
            pooled
        )

        weights = weights.view(
            -1,
            self.num_branches,
            self.channels,
        )

        weights = self.softmax(
            weights
        )

        fused = torch.zeros_like(
            features[0]
        )

        for i in range(self.num_branches):

            fused += (
                weights[:, i]
                .unsqueeze(-1)
                * features[i]
            )

        return fused, weights


def main() -> None:

    gate = AdaptiveGate(
        channels=64,
    )

    branches = [
        torch.randn(8, 64, 180),
        torch.randn(8, 64, 180),
        torch.randn(8, 64, 180),
    ]

    fused, weights = gate(branches)

    logger.info(
        "Fused Shape : %s",
        tuple(fused.shape),
    )

    logger.info(
        "Gate Shape : %s",
        tuple(weights.shape),
    )

    logger.info(
        "Sample Weights (Channel 0): %s",
        weights[0, :, 0]
        .detach()
        .cpu()
        .numpy(),
    )


if __name__ == "__main__":
    main()
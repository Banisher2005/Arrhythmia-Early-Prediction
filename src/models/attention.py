"""
Temporal Attention Module

Learns which temporal features produced by the BiLSTM are most
important for heartbeat classification.
"""

import torch
import torch.nn as nn

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TemporalAttention(nn.Module):
    """
    Temporal Attention.

    Input:
        (B, L, H)

    Output:
        context_vector : (B, H)
        attention_weights : (B, L)
    """

    def __init__(
        self,
        hidden_size: int = 256,
    ) -> None:
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(
                hidden_size,
                hidden_size,
            ),
            nn.Tanh(),
            nn.Linear(
                hidden_size,
                1,
                bias=False,
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
        x: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Parameters
        ----------
        x
            Shape:
                (B, L, H)

        Returns
        -------
        context_vector
            Shape:
                (B, H)

        attention_weights
            Shape:
                (B, L)
        """

        scores = self.attention(
            x
        )

        weights = self.softmax(
            scores
        )

        context = torch.sum(
            weights * x,
            dim=1,
        )

        weights = weights.squeeze(-1)

        return context, weights


def main() -> None:
    """
    Example usage.
    """

    model = TemporalAttention()

    x = torch.randn(
        8,
        45,
        256,
    )

    context, weights = model(x)

    logger.info(
        "Input Shape : %s",
        tuple(x.shape),
    )

    logger.info(
        "Context Shape : %s",
        tuple(context.shape),
    )

    logger.info(
        "Attention Shape : %s",
        tuple(weights.shape),
    )

    logger.info(
        "First Sample Sum : %.6f",
        weights[0].sum().item(),
    )


if __name__ == "__main__":
    main()
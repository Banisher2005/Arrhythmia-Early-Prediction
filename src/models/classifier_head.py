"""
Classification Head

Final classification layer for AMSRAN-GF.
"""

import torch
import torch.nn as nn

from src.configs import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ClassifierHead(nn.Module):
    """
    Fully connected classification head.

    Input:
        (B, 256)

    Output:
        (B, NUM_CLASSES)
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        num_classes: int = config.NUM_CLASSES,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim,
                num_classes,
            ),
        )

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
    ) -> torch.Tensor:
        """
        Forward pass.
        """

        return self.classifier(x)


def main() -> None:
    """
    Example usage.
    """

    logger.info("=" * 60)
    logger.info("Testing Classifier Head")
    logger.info("=" * 60)

    model = ClassifierHead()

    x = torch.randn(
        8,
        256,
    )

    logits = model(x)

    logger.info(
        "Input Shape  : %s",
        tuple(x.shape),
    )

    logger.info(
        "Output Shape : %s",
        tuple(logits.shape),
    )

    logger.info(
        "Output dtype : %s",
        logits.dtype,
    )

    logger.info(
        "Sample Logits: %s",
        logits[0].detach().cpu().numpy(),
    )

    logger.info("=" * 60)
    logger.info("Classifier Head Test Passed")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
"""
AMSRAN-GF

Adaptive Multi-Scale Residual Attention Network
with Gated Feature Fusion.
"""

from typing import Dict

import torch
import torch.nn as nn

from src.models.attention import TemporalAttention
from src.models.bilstm import BiLSTM
from src.models.classifier_head import ClassifierHead
from src.models.residual_cnn import ResidualCNN
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AMSRAN_GF(nn.Module):
    """
    Adaptive Multi-Scale Residual Attention Network
    with Gated Feature Fusion.

    ECG
      ↓
    Residual CNN
      ↓
    BiLSTM
      ↓
    Temporal Attention
      ↓
    Classifier
    """

    def __init__(
        self,
        input_channels: int = 1,
        cnn_feature_dim: int = 64,
        lstm_hidden_size: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()

        self.feature_extractor = ResidualCNN(
            input_channels=input_channels,
            feature_dim=cnn_feature_dim,
            dropout=dropout,
        )

        self.sequence_model = BiLSTM(
            input_size=cnn_feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            dropout=dropout,
        )

        self.attention = TemporalAttention(
            hidden_size=lstm_hidden_size * 2,
        )

        self.classifier = ClassifierHead(
            input_dim=lstm_hidden_size * 2,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        """

        cnn_features, gate_weights = self.feature_extractor(x)

        lstm_features = self.sequence_model(
            cnn_features
        )

        context_vector, attention_weights = self.attention(
            lstm_features
        )

        logits = self.classifier(
            context_vector
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        return {
            "logits": logits,
            "probabilities": probabilities,
            "cnn_features": cnn_features,
            "lstm_features": lstm_features,
            "context_vector": context_vector,
            "attention_weights": attention_weights,
            "gate_weights": gate_weights,
        }


def main() -> None:
    """
    Example usage.
    """

    logger.info("=" * 70)
    logger.info("Testing AMSRAN-GF")
    logger.info("=" * 70)

    model = AMSRAN_GF()

    x = torch.randn(
        8,
        1,
        180,
    )

    outputs = model(x)

    logger.info(
        "Logits Shape : %s",
        tuple(outputs["logits"].shape),
    )

    logger.info(
        "Probability Shape : %s",
        tuple(outputs["probabilities"].shape),
    )

    logger.info(
        "CNN Features : %s",
        tuple(outputs["cnn_features"].shape),
    )

    logger.info(
        "BiLSTM Features : %s",
        tuple(outputs["lstm_features"].shape),
    )

    logger.info(
        "Context Vector : %s",
        tuple(outputs["context_vector"].shape),
    )

    logger.info(
        "Attention Shape : %s",
        tuple(outputs["attention_weights"].shape),
    )

    logger.info("")

    for idx, gate in enumerate(
        outputs["gate_weights"],
        start=1,
    ):
        logger.info(
            "Gate %d Shape : %s",
            idx,
            tuple(gate.shape),
        )

        logger.info(
            "Gate %d Channel-0 Weights : %s",
            idx,
            gate[0, :, 0]
            .detach()
            .cpu()
            .numpy(),
        )

    logger.info("")

    prediction = torch.argmax(
        outputs["probabilities"],
        dim=1,
    )

    logger.info(
        "Prediction : %s",
        prediction,
    )

    logger.info("=" * 70)
    logger.info("AMSRAN-GF Test Passed")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
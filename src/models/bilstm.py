"""
Bidirectional LSTM Module

Models temporal dependencies in ECG feature sequences.
"""

import torch
import torch.nn as nn

from src.utils.logger import get_logger

logger = get_logger(__name__)


class BiLSTM(nn.Module):
    """
    Bidirectional LSTM Feature Extractor.

    Input:
        (B, L, C)

    Output:
        (B, L, 2 * hidden_size)
    """

    def __init__(
        self,
        input_size: int = 64,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Xavier initialization for input weights and
        orthogonal initialization for recurrent weights.
        """

        for name, param in self.lstm.named_parameters():

            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

            elif "weight_hh" in name:
                nn.init.orthogonal_(param)

            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x
            Shape:
                (B, L, input_size)

        Returns
        -------
        torch.Tensor
            Shape:
                (B, L, hidden_size * 2)
        """

        output, _ = self.lstm(x)

        output = self.dropout(output)

        return output


def main() -> None:
    """
    Example usage.
    """

    model = BiLSTM()

    x = torch.randn(
        8,
        45,
        64,
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
from typing import Optional

import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, bidirectional=True, batch_first=True
        )
        self.linear = nn.Linear(hidden_dim * 2, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        x: torch.Tensor,
        hidden_states: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # input: (batch_size, sequence_length, hidden_dim * 2)

        output, (hn, cn) = self.lstm(
            x, hidden_states
        )  # (batch_size, sequence_length, hidden_dim * 2)

        output = self.linear(output)  # (batch_size, sequence_length, output_dim)
        output = self.sigmoid(output)

        return output, (hn, cn)

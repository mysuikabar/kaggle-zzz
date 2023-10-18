from typing import Optional

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler


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


class RNNModule(LightningModule):
    def __init__(self, model: nn.Module, lr: float) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss_fn = nn.BCELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(
        self,
    ) -> OptimizerLRScheduler:
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def training_step(self, batch, batch_index) -> STEP_OUTPUT:
        X, y = batch
        pred, _ = self(X)
        loss = self.loss_fn(pred, y)
        return loss

    def validation_step(self, batch, batch_index) -> None:
        X, y = batch
        pred, _ = self(X)
        loss = self.loss_fn(pred, y)
        self.log("val_loss", loss)
        return None

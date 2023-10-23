import polars as pl
import torch
from torch.utils.data import Dataset


class SleepDataset(Dataset):
    def __init__(
        self, df: pl.DataFrame, features: list[str], targets: list[str]
    ) -> None:
        self.features = features
        self.targets = targets
        self.data = [
            df_sub
            for name, df_sub in df.group_by(
                pl.col("series_id"), pl.col("timestamp").dt.date()
            )
        ]  # split into units of (series_id x date)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        df = self.data[idx]
        X = df.select(self.features)
        y = df.select(self.targets)
        return (
            torch.tensor(X.to_numpy(), dtype=torch.float32),
            torch.tensor(y.to_numpy(), dtype=torch.float32),
        )


def pad_sequence_fn(batch):
    """
    Padding to match the longest series in the batch.
    Pass to collate_fn of DataLoader.
    """
    X, y = list(zip(*batch))
    X_pad = torch.nn.utils.rnn.pad_sequence(X, batch_first=True)
    y_pad = torch.nn.utils.rnn.pad_sequence(y, batch_first=True)
    return X_pad, y_pad

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExperimentConfig:
    wandb: bool = False
    project: str = "kaggle-zzz"
    group: str = "hoge"


@dataclass
class LSTMConfig:
    hidden_dim: int = 8
    num_layers: int = 1


@dataclass
class NNTrainConfig:
    batch_size: int = 32
    epochs: int = 3
    learning_rate: float = 0.01
    patience: int = 10


@dataclass
class Config:
    hydra: Any = field(
        default_factory=lambda: {"run": {"dir": f"./outputs/{ExperimentConfig.group}"}}
    )
    experiment: ExperimentConfig = ExperimentConfig()
    series_data_path: Path = Path("~/kaggle-zzz/data/small_data/Zzzs_train.parquet")
    events_data_path: Path = Path("~/kaggle-zzz/data/raw_data/train_events.csv")
    features: list[str] = field(default_factory=lambda: ["enmo", "anglez"])
    targets: list[str] = field(
        default_factory=lambda: ["onset_heatmap", "wakeup_heatmap"]
    )
    seed: int = 42
    n_splits: int = 4
    model: LSTMConfig = LSTMConfig()
    train: NNTrainConfig = NNTrainConfig()


ROOT = Path(__file__).parents[2]


@dataclass
class InferConfig:
    test_data_path: Path = ROOT / "data/raw_data/test_series.parquet"
    output_dir: Path = ROOT / "outputs/hoge/"
    model_file_name: str = "checkpoint-fold1.ckpt"

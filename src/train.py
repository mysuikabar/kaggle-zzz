import gc

import hydra
import polars as pl
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

import wandb
from conf.config import Config
from model.lstm import LSTM, RNNModule
from processing.features import create_dataset, load_events_data, load_series_data
from processing.torch_dataset import SleepDataset, pad_sequence_fn
from util.utils import fix_seed, seed_worker

cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(version_base="1.1", config_name="config")
def main(config: Config) -> None:
    fix_seed(config.seed)

    # preprocessing
    df_series = load_series_data(config.series_data_path)
    df_events = load_events_data(config.events_data_path)
    df = create_dataset(df_series, df_events)

    del df_series, df_events
    gc.collect()

    # run cross validation
    series_id_list = df["series_id"].unique().to_numpy()
    kfold = KFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)

    for fold, (idx_tr, idx_va) in enumerate(kfold.split(series_id_list), start=1):
        if config.experiment.wandb:
            wandb.init(
                project=config.experiment.project,
                group=config.experiment.group,
                name=f"fold-{fold}",
                config=dict(OmegaConf.to_container(config)),
            )

        # dataloader
        df_tr = df.filter(pl.col("series_id").is_in(list(series_id_list[idx_tr])))
        df_va = df.filter(pl.col("series_id").is_in(list(series_id_list[idx_va])))

        g = torch.Generator()
        g.manual_seed(config.seed)

        loader_tr = DataLoader(
            dataset=SleepDataset(df_tr, config.features, config.targets),
            batch_size=config.train.batch_size,
            shuffle=True,
            collate_fn=pad_sequence_fn,
            worker_init_fn=seed_worker,
            generator=g,
        )
        loader_va = DataLoader(
            dataset=SleepDataset(df_va, config.features, config.targets),
            batch_size=config.train.batch_size,
            collate_fn=pad_sequence_fn,
            worker_init_fn=seed_worker,
            generator=g,
        )

        # model
        lstm = LSTM(
            input_dim=len(config.features),
            hidden_dim=config.model.hidden_dim,
            output_dim=len(config.targets),
            num_layers=config.model.num_layers,
        )
        model = RNNModule(model=lstm, lr=config.train.learning_rate)

        # training
        early_stopping_callback = EarlyStopping(
            monitor="val_loss", patience=config.train.patience, verbose=False
        )
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss", dirpath=".", filename=f"checkpoint-fold{fold}"
        )
        wandb_logger = WandbLogger() if config.experiment.wandb else None

        trainer = Trainer(
            max_epochs=config.train.epochs,
            callbacks=[early_stopping_callback, checkpoint_callback],
            logger=wandb_logger,
        )
        trainer.fit(model, loader_tr, loader_va)

        if config.experiment.wandb:
            wandb.finish()


if __name__ == "__main__":
    main()

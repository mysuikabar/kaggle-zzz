import gc

import hydra
import polars as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

import wandb
from model.lstm import LSTM, RNNModule
from preprocessing.features import create_dataset, load_events_data, load_series_data
from preprocessing.torch_dataset import SleepDataset, pad_sequence_fn
from utils import fix_seed, seed_worker


@hydra.main(config_path="conf", config_name="config")
def main(config):
    params = config.params
    features = config.features
    targets = config.targets
    fix_seed(params.seed)

    # preprocessing
    df_series = load_series_data(config.series_data_path)
    df_events = load_events_data(config.events_data_path)
    df = create_dataset(df_series, df_events)

    del df_series, df_events
    gc.collect()

    # run cross validation
    series_id_list = df["series_id"].unique().to_numpy()
    kfold = KFold(n_splits=params.n_splits, shuffle=True, random_state=params.seed)

    for fold, (idx_tr, idx_va) in enumerate(kfold.split(series_id_list), start=1):
        if config.wandb:
            wandb.init(
                project=config.project,
                group=config.group,
                name=f"fold-{fold}",
                config=dict(params),
            )

        # dataloader
        df_tr = df.filter(pl.col("series_id").is_in(list(series_id_list[idx_tr])))
        df_va = df.filter(pl.col("series_id").is_in(list(series_id_list[idx_va])))

        g = torch.Generator()
        g.manual_seed(params.seed)

        loader_tr = DataLoader(
            dataset=SleepDataset(df_tr, features, targets),
            batch_size=params.batch_size,
            shuffle=True,
            collate_fn=pad_sequence_fn,
            worker_init_fn=seed_worker,
            generator=g,
        )
        loader_va = DataLoader(
            dataset=SleepDataset(df_va, features, targets),
            batch_size=params.batch_size,
            collate_fn=pad_sequence_fn,
            worker_init_fn=seed_worker,
            generator=g,
        )

        # training
        lstm = LSTM(
            input_dim=len(features),
            hidden_dim=params.hidden_dim,
            output_dim=len(targets),
            num_layers=params.num_layers,
        )
        model = RNNModule(model=lstm, lr=params.learning_rate)

        early_stopping_callback = EarlyStopping(
            monitor="val_loss", patience=params.patience, verbose=False
        )
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss", dirpath=".", filename=f"checkpoint-fold{fold}"
        )
        wandb_logger = WandbLogger() if config.wandb else None

        trainer = Trainer(
            max_epochs=params.epochs,
            callbacks=[early_stopping_callback, checkpoint_callback],
            logger=wandb_logger,
        )
        trainer.fit(model, loader_tr, loader_va)

        if config.wandb:
            wandb.finish()


if __name__ == "__main__":
    main()

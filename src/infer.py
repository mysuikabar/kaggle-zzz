import pandas as pd
from omegaconf import OmegaConf

from conf.config import InferConfig
from model.lstm import LSTM, RNNModule
from preprocessing.features import load_series_data
from preprocessing.torch_dataset import SleepDataset


def main():
    # load config
    config = OmegaConf.load(InferConfig.output_dir / ".hydra/config.yaml")

    # load test data
    df_series = load_series_data(InferConfig.test_data_path)
    dataset = SleepDataset(df_series, features=config.features)

    # load trained model
    lstm = LSTM(
        input_dim=len(config.features),
        hidden_dim=config.model.hidden_dim,
        output_dim=len(config.targets),
        num_layers=config.model.num_layers,
    )
    model = RNNModule.load_from_checkpoint(
        InferConfig.output_dir / InferConfig.model_file_name,
        model=lstm,
        lr=config.train.learning_rate,
    )
    model.eval()

    # make submission.csv
    submission = pd.DataFrame(columns=["series_id", "step", "event", "score"])

    for i, X in enumerate(dataset):
        pred, _ = model(X)
        df_sub = dataset.data[i].to_pandas()
        df_sub[["onset", "wakeup"]] = pred.detach().numpy()

        for target in ["onset", "wakeup"]:
            best_score_row = df_sub.loc[
                df_sub[target].idxmax(),
                ["series_id", "step", target],  # extract the row with the highest score
            ].rename({target: "score"})
            best_score_row["event"] = target
            submission = pd.concat([submission, best_score_row.to_frame().T])

    submission["row_id"] = list(range(len(submission)))
    submission.to_csv(InferConfig.output_dir / "submission.csv", index=False)


if __name__ == "__main__":
    main()

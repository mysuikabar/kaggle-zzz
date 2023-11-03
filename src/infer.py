from omegaconf import OmegaConf

from conf.config import InferConfig
from model.lstm import LSTM, RNNModule, predict
from processing.features import load_series_data
from processing.post_processing import make_submission
from processing.torch_dataset import SleepDataset


def main():
    # load config
    config = OmegaConf.load(InferConfig.output_dir / ".hydra/config.yaml")

    # load test data
    df_series = load_series_data(InferConfig.test_data_path)
    dataset = SleepDataset(df=df_series, features=config.features)

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
    df_pred = predict(model=model, dataset=dataset)
    submission = make_submission(df_pred)
    submission.to_csv(InferConfig.output_dir / "submission.csv", index=False)


if __name__ == "__main__":
    main()

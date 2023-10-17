from pathlib import Path

import polars as pl
from scipy.stats import norm


def _gaussian_weights(window_half: int, sigma: float) -> list[float]:
    weights = norm.pdf(
        list(range(-window_half, window_half + 1)), scale=sigma
    ) / norm.pdf(0, scale=sigma)
    return weights


def load_series_data(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path).with_columns(pl.col("timestamp").str.to_datetime())


def load_events_data(path: Path) -> pl.DataFrame:
    return (
        pl.read_csv(path)
        .drop_nulls(subset="step")
        .with_columns(pl.col("step").cast(pl.UInt32))
    )


def create_dataset(
    df_series: pl.DataFrame,
    df_events: pl.DataFrame,
    window_half: int = 360,
    sigma: float = 120,
) -> pl.DataFrame:
    weights = _gaussian_weights(window_half=window_half, sigma=sigma)
    df = (
        df_series.join(df_events, on=["series_id", "step"], how="left")
        .with_columns(
            pl.when(pl.col("event") == "onset").then(1).otherwise(0).alias("onset"),
            pl.when(pl.col("event") == "wakeup").then(1).otherwise(0).alias("wakeup"),
        )
        .with_columns(
            pl.col("onset")
            .rolling_sum(
                window_size=len(weights), weights=weights, min_periods=1, center=True
            )
            .over("series_id")
            .alias("onset_heatmap"),
            pl.col("wakeup")
            .rolling_sum(
                window_size=len(weights), weights=weights, min_periods=1, center=True
            )
            .over("series_id")
            .alias("wakeup_heatmap"),
        )
    )
    return df

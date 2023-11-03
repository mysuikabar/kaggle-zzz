import pandas as pd
import polars as pl


def make_submission(df: pl.DataFrame) -> pd.DataFrame:
    submission = pd.DataFrame(columns=["series_id", "step", "event", "score"])
    df_grouped = df.group_by(pl.col("series_id"), pl.col("timestamp").dt.date())

    for _, df_sub in df_grouped:
        df_sub = df_sub.to_pandas()

        for target in ["onset", "wakeup"]:
            best_score_row = df_sub.loc[
                df_sub[target].idxmax(),
                ["series_id", "step", target],  # extract the row with the highest score
            ].rename({target: "score"})
            best_score_row["event"] = target
            submission = pd.concat([submission, best_score_row.to_frame().T])

    submission["row_id"] = list(range(len(submission)))
    submission = submission.reindex(
        ["row_id", "series_id", "step", "event", "score"], axis="columns"
    ).reset_index(drop=True)

    return submission

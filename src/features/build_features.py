from pathlib import Path
import json
import numpy as np
import pandas as pd

DATA_PATH = Path("data/processed/fd001_h30.parquet")
OUT_DIR = Path("data/processed")
ARTIFACTS_DIR = Path("artifacts")

# Sensors in C-MAPSS: s_1 ... s_21
SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]

def build_window_features(
    df: pd.DataFrame,
    window: int = 30,
    min_periods: int = 10,
) -> pd.DataFrame:
    """
    Build rolling window features for each engine_id.
    For each sensor: mean/std/min/max over last `window` cycles.

    We keep only rows where we have at least `min_periods` observations in the window.
    Target is taken from the current row (i.e., label aligned to the same cycle).
    """
    df = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)

    feats = []
    for col in SENSOR_COLS:
        g = df.groupby("engine_id")[col]
        feats.append(g.rolling(window=window, min_periods=min_periods).mean().reset_index(level=0, drop=True).rename(f"{col}_mean_w{window}"))
        feats.append(g.rolling(window=window, min_periods=min_periods).std().reset_index(level=0, drop=True).rename(f"{col}_std_w{window}"))
        feats.append(g.rolling(window=window, min_periods=min_periods).min().reset_index(level=0, drop=True).rename(f"{col}_min_w{window}"))
        feats.append(g.rolling(window=window, min_periods=min_periods).max().reset_index(level=0, drop=True).rename(f"{col}_max_w{window}"))

    X = pd.concat(feats, axis=1)

    out = pd.concat(
        [
            df[["engine_id", "cycle", "rul", "will_fail_within_horizon"]],
            X
        ],
        axis=1
    )

    # drop rows with NaNs from rolling (early cycles)
    out = out.dropna().reset_index(drop=True)
    return out

def main(window: int = 30, min_periods: int = 10):
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run: python src/data/make_dataset.py"
        )

    df = pd.read_parquet(DATA_PATH)
    features_df = build_window_features(df, window=window, min_periods=min_periods)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    out_path = OUT_DIR / f"fd001_features_w{window}.parquet"
    features_df.to_parquet(out_path, index=False)

    # Save feature column list for training/inference consistency
    feature_cols = [c for c in features_df.columns if c not in ["engine_id", "cycle", "rul", "will_fail_within_horizon"]]
    (ARTIFACTS_DIR / "feature_columns.json").write_text(json.dumps(feature_cols, indent=2))

    print(f"Saved features to: {out_path}")
    print(f"Rows: {len(features_df):,}")
    print(f"Num features: {len(feature_cols)}")
    print(features_df[["engine_id", "cycle", "rul", "will_fail_within_horizon"]].head())

if __name__ == "__main__":
    main()

from pathlib import Path
import json
import argparse
import pandas as pd

DATA_PATH = Path("data/processed/fd001_h30.parquet")
FEATURE_COLS_PATH = Path("artifacts/feature_columns.json")

SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]

def compute_last_window_features(df_engine: pd.DataFrame, window: int = 30) -> dict:
    """
    Compute rolling window stats for the LAST cycle of a given engine.
    Produces exactly the same feature names as build_features.py:
    s_i_{mean,std,min,max}_w{window}
    """
    df_engine = df_engine.sort_values("cycle").reset_index(drop=True)

    last_window = df_engine.tail(window)
    feats = {}
    for col in SENSOR_COLS:
        series = last_window[col]
        feats[f"{col}_mean_w{window}"] = float(series.mean())
        feats[f"{col}_std_w{window}"] = float(series.std(ddof=1)) if len(series) > 1 else 0.0
        feats[f"{col}_min_w{window}"] = float(series.min())
        feats[f"{col}_max_w{window}"] = float(series.max())
    return feats

def main(engine_id: int, window: int = 30, out: str = "payload.json"):
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found. Run dataset preparation first.")

    df = pd.read_parquet(DATA_PATH)
    df_engine = df[df["engine_id"] == engine_id]

    if df_engine.empty:
        raise ValueError(f"engine_id={engine_id} not found in dataset.")

    feats = compute_last_window_features(df_engine, window=window)

    # Ensure exact feature list (ordering is handled in API, but we validate presence)
    if FEATURE_COLS_PATH.exists():
        expected = set(json.loads(FEATURE_COLS_PATH.read_text()))
        got = set(feats.keys())
        missing = sorted(list(expected - got))
        extra = sorted(list(got - expected))
        if missing:
            raise ValueError(f"Missing expected features (first 10): {missing[:10]}")
        if extra:
            # extra should not happen, but keep strict
            raise ValueError(f"Unexpected extra features (first 10): {extra[:10]}")

    payload = {"engine_id": int(engine_id), "features": feats}

    Path(out).write_text(json.dumps(payload, indent=2))
    print(f"Saved payload to: {out}")
    print(json.dumps(payload, indent=2)[:800] + "\n...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-id", type=int, required=True)
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--out", type=str, default="payload.json")
    args = parser.parse_args()

    main(engine_id=args.engine_id, window=args.window, out=args.out)

from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")

# C-MAPSS columns: id, cycle, 3 operational settings, 21 sensors
COLS = (
    ["engine_id", "cycle"]
    + [f"op_{i}" for i in range(1, 4)]
    + [f"s_{i}" for i in range(1, 22)]
)

def load_train_fd001(path: Path) -> pd.DataFrame:
    # NASA files are space-separated with trailing spaces
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, :len(COLS)]
    df.columns = COLS
    return df

def add_rul_and_label(df: pd.DataFrame, horizon: int = 30) -> pd.DataFrame:
    # RUL = max_cycle(engine) - current_cycle
    max_cycle = df.groupby("engine_id")["cycle"].max().rename("max_cycle")
    df = df.merge(max_cycle, on="engine_id", how="left")
    df["rul"] = df["max_cycle"] - df["cycle"]
    df["will_fail_within_horizon"] = (df["rul"] <= horizon).astype(int)
    df = df.drop(columns=["max_cycle"])
    return df

def main(horizon: int = 30):
    train_path = DATA_DIR / "train_FD001.txt"
    if not train_path.exists():
        raise FileNotFoundError(
            "train_FD001.txt not found. Run: python src/data/download_cmapps.py"
        )

    df = load_train_fd001(train_path)
    df = add_rul_and_label(df, horizon=horizon)

    out_dir = DATA_DIR / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"fd001_h{horizon}.parquet"
    df.to_parquet(out_path, index=False)

    print(f"Saved processed dataset to: {out_path}")
    print(df[["engine_id", "cycle", "rul", "will_fail_within_horizon"]].head())

if __name__ == "__main__":
    main()

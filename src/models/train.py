from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, confusion_matrix

FEATURES_PATH = Path("data/processed/fd001_features_w30.parquet")
ARTIFACTS_DIR = Path("artifacts")
FEATURE_COLS_PATH = ARTIFACTS_DIR / "feature_columns.json"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

TARGET_COL = "will_fail_within_horizon"
GROUP_COL = "engine_id"

def load_feature_columns(df: pd.DataFrame) -> list[str]:
    if FEATURE_COLS_PATH.exists():
        return json.loads(FEATURE_COLS_PATH.read_text())
    # fallback: infer
    return [c for c in df.columns if c not in ["engine_id", "cycle", "rul", TARGET_COL]]

def group_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split by engine_id to avoid leakage across time for the same engine.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    groups = df[GROUP_COL].values
    idx_train, idx_test = next(gss.split(df, groups=groups))
    return df.iloc[idx_train].reset_index(drop=True), df.iloc[idx_test].reset_index(drop=True)

def main():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"{FEATURES_PATH} not found. Run: python src/features/build_features.py"
        )

    df = pd.read_parquet(FEATURES_PATH)

    train_df, test_df = group_split(df, test_size=0.2, random_state=42)

    feature_cols = load_feature_columns(df)

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL].astype(int)

    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL].astype(int)

    # Baseline model: Logistic Regression with scaling
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                n_jobs=None,
                random_state=42
            )),
        ]
    )

    pipe.fit(X_train, y_train)

    # Probabilities for ROC-AUC
    y_prob = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "f1": float(f1_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "test_size_engines": int(test_df[GROUP_COL].nunique()),
        "train_size_engines": int(train_df[GROUP_COL].nunique()),
        "num_features": int(len(feature_cols)),
        "threshold": 0.5,
        "model": "LogisticRegression + StandardScaler",
        "window": 30,
        "horizon": 30,
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()

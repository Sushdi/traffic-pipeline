"""
Train multiple classifiers on the 15-minute-ahead traffic prediction task.
The model learns to predict congestion_level at time T+15min using features
collected at time T. Tracks experiments with MLflow, saves the best model.
"""

import json
import logging
import os
import warnings

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for CI
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

with open("params.yaml") as f:
    CFG = yaml.safe_load(f)

PROC_PATH  = CFG["paths"]["processed"]
MODEL_PATH = CFG["paths"]["model"]
TEST_SIZE  = CFG["training"]["test_size"]
SEED       = CFG["training"]["random_state"]
N_EST      = CFG["training"]["n_estimators"]

FEATURE_COLS = [
    # Cyclical time
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "min_sin", "min_cos",
    # Peak flags
    "is_weekend", "is_peak_morning", "is_peak_evening",
    # Raw flow only — NO ratios, NO ratio lags
    "current_speed", "free_flow_speed",
    "current_travel_time", "free_flow_travel_time",
    "confidence", "road_closure", "frc_code",
    # Speed lags only
    "speed_lag_1", "speed_lag_3", "speed_lag_6", "speed_lag_12",
    # Rolling
    "speed_roll_mean_6", "speed_roll_mean_12", "speed_roll_std_6",
    "tt_ratio_roll_6", "speed_trend",
]

# Model trained to predict 15 minutes ahead (3 steps × 5-min interval)
TARGET = "target_15m"

MODELS = {
    "random_forest": RandomForestClassifier(
        n_estimators=N_EST, max_depth=12,
        min_samples_leaf=4, random_state=SEED, n_jobs=-1,
    ),
    "xgboost": XGBClassifier(
        n_estimators=N_EST, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss", random_state=SEED,
    ),
    "lightgbm": LGBMClassifier(
        n_estimators=N_EST, max_depth=8, learning_rate=0.05,
        num_leaves=63, random_state=SEED, verbose=-1,
    ),
}


def plot_confusion_matrix(clf, X_test, y_test, le, run_dir: str, name: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        clf, X_test, y_test,
        display_labels=le.classes_,
        cmap="Blues", ax=ax,
    )
    ax.set_title(f"Confusion matrix — {name}")
    path = os.path.join(run_dir, "confusion_matrix.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_feature_importance(clf, feature_names: list, run_dir: str, name: str):
    if not hasattr(clf, "feature_importances_"):
        return None
    imp = clf.feature_importances_
    idx = np.argsort(imp)[-20:]   # top 20
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh([feature_names[i] for i in idx], imp[idx])
    ax.set_title(f"Feature importance — {name}")
    ax.set_xlabel("Importance")
    path = os.path.join(run_dir, "feature_importance.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def train():
    df = pd.read_csv(PROC_PATH)

    # Validate all feature columns exist
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[FEATURE_COLS].values
    le = LabelEncoder()
    # Fit encoder on the FUTURE target labels (same classes, but explicit)
    y = le.fit_transform(df[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED,
    )

    log.info("Train: %d  Test: %d  Classes: %s", len(X_train), len(X_test), list(le.classes_))
    mlflow.set_tracking_uri("file:///tmp/mlruns")
    mlflow.set_experiment("traffic-congestion-munich")

    best = {"f1": 0.0, "model": None, "name": "", "le": le}
    all_metrics = {}

    for name, clf in MODELS.items():
        log.info("\n=== %s ===", name)
        run_dir = f"mlruns/artifacts/{name}"
        os.makedirs(run_dir, exist_ok=True)

        with mlflow.start_run(run_name=name):
            # 5-fold CV on training set
            cv_scores = cross_val_score(
                clf, X_train, y_train,
                cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                scoring="f1_weighted", n_jobs=-1,
            )
            log.info("CV F1: %.4f ± %.4f", cv_scores.mean(), cv_scores.std())

            clf.fit(X_train, y_train)
            y_pred  = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)

            f1  = f1_score(y_test, y_pred, average="weighted")
            ll  = log_loss(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")

            # Log params + metrics
            mlflow.log_params({"model": name, "n_estimators": N_EST, "seed": SEED})
            mlflow.log_metrics({
                "cv_f1_mean":   float(cv_scores.mean()),
                "cv_f1_std":    float(cv_scores.std()),
                "test_f1":      float(f1),
                "test_log_loss": float(ll),
                "test_roc_auc": float(auc),
            })

            # Artifacts
            cm_path = plot_confusion_matrix(clf, X_test, y_test, le, run_dir, name)
            fi_path = plot_feature_importance(clf, FEATURE_COLS, run_dir, name)
            mlflow.log_artifact(cm_path)
            if fi_path:
                mlflow.log_artifact(fi_path)

            mlflow.sklearn.log_model(clf, "model")

            log.info(classification_report(y_test, y_pred, target_names=le.classes_))
            log.info("F1=%.4f  LogLoss=%.4f  AUC=%.4f", f1, ll, auc)

            all_metrics[name] = {"f1": f1, "log_loss": ll, "roc_auc": auc}

            if f1 > best["f1"]:
                best.update({"f1": f1, "model": clf, "name": name, "le": le})

    # Save best model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    bundle = {
        "model":              best["model"],
        "label_encoder":      best["le"],
        "features":           FEATURE_COLS,
        "best_model":         best["name"],
        "best_f1":            best["f1"],
        "target":             TARGET,           # "target_15m"
        "horizon_minutes":    15,               # what the model predicts ahead
        "interval_seconds":   CFG["collection"]["interval_seconds"],
    }
    joblib.dump(bundle, MODEL_PATH)
    log.info("\n✓ Best: %s (F1=%.4f) → %s", best["name"], best["f1"], MODEL_PATH)

    # Write DVC metrics file
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/scores.json", "w") as fh:
        json.dump({
            "best_model": best["name"],
            "best_f1":    round(best["f1"], 4),
            "all":        all_metrics,
        }, fh, indent=2)


if __name__ == "__main__":
    train()
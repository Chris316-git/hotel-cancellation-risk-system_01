"""Compare several model families on the same temporal split.

Always runs: logistic regression, HistGradientBoosting, random forest (scikit-learn).
Runs if installed: LightGBM, XGBoost.

Run: python -m hotel_risk.compare_models data/raw/hotel_bookings.csv
Writes reports/model_comparison.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from .data import CATEGORICAL, NUMERIC, TARGET, booking_time_features, clean, load_raw, temporal_split
from .evaluate import score
from .models import make_baseline, make_gbm


def _ordinal_pre() -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                               encoded_missing_value=-1, min_frequency=50), CATEGORICAL),
        ("num", "passthrough", NUMERIC)])


def candidates() -> dict:
    out = {
        "logistic_regression": make_baseline,
        "hist_gbm": make_gbm,
        "random_forest": lambda: Pipeline([("pre", _ordinal_pre()), ("clf", RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=0))]),
    }
    try:
        from lightgbm import LGBMClassifier
        out["lightgbm"] = lambda: Pipeline([("pre", _ordinal_pre()), ("clf", LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31, random_state=0, verbose=-1))])
    except Exception as e:  # not installed, or a native library (e.g. libomp on macOS) is missing
        print(f"skipping optional model: {e.__class__.__name__}")
    try:
        from xgboost import XGBClassifier
        out["xgboost"] = lambda: Pipeline([("pre", _ordinal_pre()), ("clf", XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=6, random_state=0, eval_metric="logloss"))])
    except Exception as e:  # not installed, or a native library (e.g. libomp on macOS) is missing
        print(f"skipping optional model: {e.__class__.__name__}")
    return out


def run(path: str) -> dict:
    df = clean(load_raw(path))
    train, test = temporal_split(df)
    Xtr, Xte = booking_time_features(train), booking_time_features(test)
    res = {}
    for name, factory in candidates().items():
        t0 = time.time()
        p = factory().fit(Xtr, train[TARGET]).predict_proba(Xte)[:, 1]
        res[name] = {**score(test[TARGET], p), "fit_seconds": round(time.time() - t0, 1)}
    return res


if __name__ == "__main__":
    out = run(sys.argv[1] if len(sys.argv) > 1 else "data/raw/hotel_bookings.csv")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/model_comparison.json").write_text(json.dumps(out, indent=2))
    for k, v in out.items():
        print(f"{k:22s}", v)

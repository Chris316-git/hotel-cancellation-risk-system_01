"""Phase 2 evaluation: temporal vs random split, baseline vs GBM.

Run: python -m hotel_risk.evaluate data/raw/hotel_bookings.csv
Writes reports/phase2_metrics.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from .data import TARGET, booking_time_features, clean, load_raw, temporal_split
from .models import make_baseline, make_gbm


def score(y, p) -> dict:
    return {
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "log_loss": round(float(log_loss(y, p)), 4),
        "base_rate": round(float(np.mean(y)), 4),
    }


def run(path: str) -> dict:
    df = clean(load_raw(path))
    out: dict = {}
    models = {"logistic_baseline": make_baseline, "hist_gbm": make_gbm}

    train, test = temporal_split(df)
    for name, factory in models.items():
        m = factory().fit(booking_time_features(train), train[TARGET])
        out[f"temporal/{name}"] = score(test[TARGET], m.predict_proba(booking_time_features(test))[:, 1])

    # Random split on the same data: shows how optimistic non-temporal evaluation is.
    tr, te = train_test_split(df, test_size=len(test) / len(df), random_state=0, stratify=df[TARGET])
    for name, factory in models.items():
        m = factory().fit(booking_time_features(tr), tr[TARGET])
        out[f"random/{name}"] = score(te[TARGET], m.predict_proba(booking_time_features(te))[:, 1])
    return out


if __name__ == "__main__":
    res = run(sys.argv[1] if len(sys.argv) > 1 else "data/raw/hotel_bookings.csv")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/phase2_metrics.json").write_text(json.dumps(res, indent=2))
    for k, v in res.items():
        print(f"{k:28s}", v)

"""Phase 3: probability calibration and decision-threshold selection.

Three chronological slices (no shuffling):
  train  : arrivals < 2016-12-01      -> fit the model
  calib  : 2016-12-01 .. 2017-02-28   -> fit calibrator, choose threshold
  test   : arrivals >= 2017-03-01     -> untouched final evaluation

Run: python -m hotel_risk.calibrate data/raw/hotel_bookings.csv
Writes reports/phase3_calibration.json and reports/reliability.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .data import TARGET, booking_time_features, clean, load_raw
from .models import make_gbm

TRAIN_END = pd.Timestamp("2016-12-01")
CALIB_END = pd.Timestamp("2017-03-01")


def split3(df: pd.DataFrame):
    a = df["arrival_date"]
    return df[a < TRAIN_END], df[(a >= TRAIN_END) & (a < CALIB_END)], df[a >= CALIB_END]


def ece(y, p, bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    y, p = np.asarray(y), np.asarray(p)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def metrics(y, p) -> dict:
    return {"roc_auc": round(float(roc_auc_score(y, p)), 4), "brier": round(float(brier_score_loss(y, p)), 4),
            "log_loss": round(float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))), 4), "ece": round(ece(y, p), 4)}


def pick_threshold(y, p, min_precision: float = 0.80) -> float:
    """Lowest threshold whose precision on the calibration slice is >= min_precision
    (i.e. flag as many risky bookings as possible while keeping flags trustworthy)."""
    best = 0.5
    for t in np.linspace(0.3, 0.95, 66):
        flagged = p >= t
        if flagged.sum() >= 50 and y[flagged].mean() >= min_precision:
            return float(t)
        best = float(t)
    return best


def at_threshold(y, p, t) -> dict:
    flag = p >= t
    tp = int((flag & (y == 1)).sum())
    return {"threshold": round(float(t), 3), "flag_rate": round(float(flag.mean()), 4),
            "precision": round(tp / max(int(flag.sum()), 1), 4),
            "recall": round(tp / max(int((y == 1).sum()), 1), 4)}


def run(path: str) -> dict:
    df = clean(load_raw(path))
    train, calib, test = split3(df)
    X = booking_time_features
    model = make_gbm().fit(X(train), train[TARGET])
    p_cal_raw = model.predict_proba(X(calib))[:, 1]
    p_test_raw = model.predict_proba(X(test))[:, 1]
    y_cal, y_test = calib[TARGET].to_numpy(), test[TARGET].to_numpy()

    # Platt scaling (sigmoid on the logit) and isotonic regression, fitted on the calibration slice.
    logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    platt = LogisticRegression(C=1e6).fit(logit(p_cal_raw).reshape(-1, 1), y_cal)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_cal_raw, y_cal)
    versions = {
        "raw": p_test_raw,
        "platt": platt.predict_proba(logit(p_test_raw).reshape(-1, 1))[:, 1],
        "isotonic": iso.predict(p_test_raw),
    }
    out = {"sizes": {"train": len(train), "calib": len(calib), "test": len(test)},
           "base_rate": {"calib": round(float(y_cal.mean()), 4), "test": round(float(y_test.mean()), 4)},
           "test_metrics": {k: metrics(y_test, v) for k, v in versions.items()}}

    # Threshold chosen on calibration slice (raw scores), applied unchanged to test.
    t = pick_threshold(y_cal, p_cal_raw)
    out["threshold_policy"] = "lowest threshold with >=80% precision on calibration slice"
    out["threshold_at_test"] = at_threshold(y_test, p_test_raw, t)
    out["top_risk_buckets_test"] = {
        f"top_{int(q * 100)}pct": {"precision": round(float(y_test[p_test_raw >= np.quantile(p_test_raw, 1 - q)].mean()), 4)}
        for q in (0.1, 0.2, 0.3)}

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
        for k, v in versions.items():
            fr, mp = calibration_curve(y_test, v, n_bins=10, strategy="quantile")
            ax.plot(mp, fr, marker="o", label=k)
        ax.set(xlabel="Predicted cancel probability", ylabel="Observed cancel rate", title="Reliability (test, 2017-03+)")
        ax.legend(); fig.tight_layout()
        Path("reports").mkdir(exist_ok=True)
        fig.savefig("reports/reliability.png", dpi=150)
    except Exception as e:  # plotting is optional
        out["plot_error"] = str(e)
    return out


if __name__ == "__main__":
    res = run(sys.argv[1] if len(sys.argv) > 1 else "data/raw/hotel_bookings.csv")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/phase3_calibration.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))

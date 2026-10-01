"""Train the production bundle: python -m hotel_risk.train data/raw/hotel_bookings.csv models/model.joblib

Uses the validated recipe from Phase 3: GBM fitted on the earliest slice, isotonic calibrator on the next.
"""
from __future__ import annotations

import sys

from sklearn.isotonic import IsotonicRegression

from .calibrate import split3
from .data import TARGET, booking_time_features, clean, load_raw
from .impact import Assumptions
from .models import make_gbm
from .predict import Bundle, save_bundle


def build_bundle(df) -> Bundle:
    train, calib, _ = split3(df)
    model = make_gbm().fit(booking_time_features(train), train[TARGET])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(
        model.predict_proba(booking_time_features(calib))[:, 1], calib[TARGET])
    return Bundle(model, iso, Assumptions(), str(calib["arrival_date"].max().date()))


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "data/raw/hotel_bookings.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "models/model.joblib"
    save_bundle(build_bundle(clean(load_raw(src))), out)
    print(f"saved {out}")

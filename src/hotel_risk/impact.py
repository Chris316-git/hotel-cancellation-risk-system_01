"""Phase 4: business-impact layer. Turns cancellation risk into an overbooking decision.

Policy: for each booking, the hotel may "oversell" that room (accept another booking on top of it).
  * If the booking then cancels  -> gain  G = resell_fill_rate * adr * nights   (room re-filled)
  * If the guest shows up        -> cost  C = walk_cost_nights * adr + walk_fixed (relocate guest elsewhere)
Flagging is worth it when  p*G > (1-p)*C,  i.e. p > C / (G + C)  (a per-booking threshold from calibrated p).

ASSUMPTIONS (illustrative, editable): see Assumptions. Nothing here is hotel-specific data.
Run: python -m hotel_risk.impact data/raw/hotel_bookings.csv   -> reports/phase4_impact.json
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

from .calibrate import split3
from .data import TARGET, booking_time_features, clean, load_raw
from .models import make_gbm


@dataclass
class Assumptions:
    resell_fill_rate: float = 0.5   # chance a freed room is actually re-sold
    walk_cost_nights: float = 2.0   # compensation + alternative hotel, in multiples of the nightly rate
    walk_fixed: float = 50.0        # fixed goodwill cost per relocated guest


def value(flag: np.ndarray, y: np.ndarray, adr: np.ndarray, nights: np.ndarray, a: Assumptions) -> float:
    gain = a.resell_fill_rate * adr * nights
    cost = a.walk_cost_nights * adr + a.walk_fixed
    return float(np.sum(np.where(flag & (y == 1), gain, 0) - np.where(flag & (y == 0), cost, 0)))


def run(path: str) -> dict:
    df = clean(load_raw(path))
    train, calib, test = split3(df)
    X = booking_time_features
    model = make_gbm().fit(X(train), train[TARGET])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(
        model.predict_proba(X(calib))[:, 1], calib[TARGET])
    p = iso.predict(model.predict_proba(X(test))[:, 1])
    y = test[TARGET].to_numpy()
    adr = test["adr"].to_numpy()
    nights = np.maximum(test["total_nights"].to_numpy(), 1)

    def policies(a: Assumptions) -> dict:
        gain = a.resell_fill_rate * adr * nights
        cost = a.walk_cost_nights * adr + a.walk_fixed
        res = {
            "no_overbooking": value(np.zeros(len(y), bool), y, adr, nights, a),
            "fixed_threshold_0.5": value(p >= 0.5, y, adr, nights, a),
            "fixed_threshold_0.8": value(p >= 0.8, y, adr, nights, a),
            "cost_aware_model": value(p > cost / (gain + cost + 1e-9), y, adr, nights, a),
            "oracle_upper_bound": value(y == 1, y, adr, nights, a),
        }
        return {k: round(v) for k, v in res.items()}

    base = Assumptions()
    out = {"assumptions": asdict(base), "test_bookings": len(y), "net_value_usd": policies(base)}
    best = out["net_value_usd"]
    out["captured_share_of_oracle"] = round(best["cost_aware_model"] / best["oracle_upper_bound"], 3)
    out["sensitivity_cost_aware_net_value_usd"] = {
        f"fill={f}, walk={w}x": policies(Assumptions(f, w))["cost_aware_model"]
        for f in (0.3, 0.5, 0.8) for w in (1.0, 2.0, 4.0)}
    return out


if __name__ == "__main__":
    res = run(sys.argv[1] if len(sys.argv) > 1 else "data/raw/hotel_bookings.csv")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/phase4_impact.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))

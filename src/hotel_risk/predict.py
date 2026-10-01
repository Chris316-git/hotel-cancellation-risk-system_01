"""Inference: score booking-time records and turn risk into an overbooking decision."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import booking_time_features, clean
from .impact import Assumptions


@dataclass
class Bundle:
    model: object          # fitted sklearn pipeline (raw probabilities)
    calibrator: object     # fitted isotonic regression
    assumptions: Assumptions
    trained_through: str   # last arrival date seen in training/calibration data


def save_bundle(bundle: Bundle, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: str | Path) -> Bundle:
    return joblib.load(path)


def score(bundle: Bundle, records: list[dict]) -> pd.DataFrame:
    """records: raw booking-time fields (no outcome columns). Returns risk + decision per record."""
    raw = pd.DataFrame(records)
    df = clean(raw)
    if len(df) != len(raw):
        raise ValueError("Some records are invalid (no guests or negative adr).")
    p = bundle.calibrator.predict(bundle.model.predict_proba(booking_time_features(df))[:, 1])
    a = bundle.assumptions
    nights = np.maximum(df["total_nights"].to_numpy(), 1)
    gain = a.resell_fill_rate * df["adr"].to_numpy() * nights
    cost = a.walk_cost_nights * df["adr"].to_numpy() + a.walk_fixed
    return pd.DataFrame({
        "cancel_probability": np.round(p, 4),
        "risk_band": pd.cut(p, [-0.01, 0.25, 0.6, 1.0], labels=["low", "medium", "high"]).astype(str),
        "oversell_recommended": p > cost / (gain + cost + 1e-9),
    })

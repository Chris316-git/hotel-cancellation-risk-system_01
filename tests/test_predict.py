import numpy as np
import pandas as pd

from hotel_risk.data import clean
from hotel_risk.predict import load_bundle, save_bundle, score
from hotel_risk.train import build_bundle

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]


def synthetic(n=1500, seed=0):
    r = np.random.default_rng(seed)
    year = r.choice([2015, 2016, 2017], n, p=[.3, .5, .2])
    lead = r.integers(0, 300, n)
    df = pd.DataFrame({
        "hotel": r.choice(["City Hotel", "Resort Hotel"], n), "lead_time": lead,
        "arrival_date_year": year, "arrival_date_month": r.choice(MONTHS[:8], n),
        "arrival_date_week_number": r.integers(1, 53, n), "arrival_date_day_of_month": r.integers(1, 28, n),
        "stays_in_weekend_nights": r.integers(0, 3, n), "stays_in_week_nights": r.integers(0, 5, n),
        "adults": r.integers(1, 3, n), "children": 0, "babies": 0, "meal": "BB",
        "country": r.choice(["PRT", "GBR", None], n), "market_segment": r.choice(["Online TA", "Direct"], n),
        "distribution_channel": "TA/TO", "is_repeated_guest": 0, "previous_cancellations": 0,
        "previous_bookings_not_canceled": 0, "reserved_room_type": "A",
        "deposit_type": r.choice(["No Deposit", "Non Refund"], n), "agent": r.choice([9.0, 1.0, None], n),
        "company": None, "customer_type": "Transient", "adr": r.uniform(40, 200, n),
    })
    df["is_canceled"] = (r.random(n) < 0.15 + lead / 600).astype(int)
    return df


def test_bundle_roundtrip_and_score(tmp_path):
    df = synthetic()
    full = df.assign(assigned_room_type="A", booking_changes=0, days_in_waiting_list=0,
                     required_car_parking_spaces=0, total_of_special_requests=0,
                     reservation_status="x", reservation_status_date="2017-01-01")
    bundle = build_bundle(clean(full))
    path = tmp_path / "m.joblib"
    save_bundle(bundle, path)
    out = score(load_bundle(path), df.drop(columns="is_canceled").head(5).to_dict("records"))
    assert len(out) == 5
    assert out["cancel_probability"].between(0, 1).all()
    assert set(out["risk_band"]) <= {"low", "medium", "high"}


def test_invalid_record_rejected(tmp_path):
    import pytest
    df = synthetic()
    full = df.assign(assigned_room_type="A", booking_changes=0, days_in_waiting_list=0,
                     required_car_parking_spaces=0, total_of_special_requests=0,
                     reservation_status="x", reservation_status_date="2017-01-01")
    bundle = build_bundle(clean(full))
    bad = df.drop(columns="is_canceled").head(1).assign(adults=0).to_dict("records")
    with pytest.raises(ValueError):
        score(bundle, bad)

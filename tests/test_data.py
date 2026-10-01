import pandas as pd

from hotel_risk.data import (CATEGORICAL, LEAKAGE_COLUMNS, NUMERIC, TARGET,
                             booking_time_features, clean, temporal_split)


def _toy():
    return pd.DataFrame({
        "hotel": ["City Hotel"] * 3, "is_canceled": [0, 1, 0], "lead_time": [10, 20, 30],
        "arrival_date_year": [2016, 2016, 2017], "arrival_date_month": ["May", "June", "April"],
        "arrival_date_week_number": [20, 24, 15], "arrival_date_day_of_month": [10, 5, 3],
        "stays_in_weekend_nights": [1, 0, 2], "stays_in_week_nights": [2, 3, 1],
        "adults": [2, 2, 0], "children": [0, None, 0], "babies": [0, 0, 0],
        "meal": ["BB"] * 3, "country": ["PRT", None, "GBR"], "market_segment": ["Direct"] * 3,
        "distribution_channel": ["Direct"] * 3, "is_repeated_guest": [0] * 3,
        "previous_cancellations": [0] * 3, "previous_bookings_not_canceled": [0] * 3,
        "reserved_room_type": ["A"] * 3, "assigned_room_type": ["A"] * 3,
        "booking_changes": [0] * 3, "deposit_type": ["No Deposit"] * 3, "agent": [9, None, 1],
        "company": [None] * 3, "days_in_waiting_list": [0] * 3, "customer_type": ["Transient"] * 3,
        "adr": [100.0, 90.0, 80.0], "required_car_parking_spaces": [0] * 3,
        "total_of_special_requests": [0] * 3, "reservation_status": ["Check-Out", "Canceled", "Check-Out"],
        "reservation_status_date": ["2016-05-13"] * 3,
    })


def test_clean_drops_zero_guest_rows():
    assert len(clean(_toy())) == 2


def test_features_exclude_leakage_and_target():
    X = booking_time_features(clean(_toy()))
    assert not set(X.columns) & set(LEAKAGE_COLUMNS)
    assert TARGET not in X.columns
    assert list(X.columns) == CATEGORICAL + NUMERIC


def test_temporal_split_has_no_overlap_in_time():
    df = clean(_toy())
    tr, te = temporal_split(df, "2016-06-01")
    assert tr["arrival_date"].max() < te["arrival_date"].min()

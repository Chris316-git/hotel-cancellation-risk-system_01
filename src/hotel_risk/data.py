"""Data loading, cleaning and leakage-safe feature selection (Phase 1)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET = "is_canceled"

MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], start=1)}

# Columns that are NOT known when the booking is made. Using them would leak
# the outcome, so they are excluded from the modelling feature set.
LEAKAGE_COLUMNS = {
    "reservation_status": "Final outcome (Canceled / Check-Out / No-Show); the target in disguise.",
    "reservation_status_date": "Date of the final status; reveals the outcome.",
    "assigned_room_type": "Assigned at/near check-in, after cancellations are known.",
    "booking_changes": "Counts changes made AFTER booking; accumulates over time.",
    "days_in_waiting_list": "Often finalised after the booking outcome; excluded to be strict.",
    "required_car_parking_spaces": "Typically requested/updated after booking; canceled bookings are all 0.",
    "total_of_special_requests": "Can be added after booking; excluded for strict booking-time scoring.",
}

CATEGORICAL = [
    "hotel", "meal", "country", "market_segment", "distribution_channel",
    "reserved_room_type", "deposit_type", "customer_type", "agent", "company",
]
NUMERIC = [
    "lead_time", "arrival_date_week_number", "arrival_date_day_of_month",
    "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children",
    "babies", "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "adr",
    "total_nights", "total_guests", "arrival_month", "arrival_dow", "has_agent", "has_company",
]


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the CSV, treating the literal string 'NULL' as missing."""
    return pd.read_csv(path, na_values=["NULL"])


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning + booking-time derived features. Returns a new frame."""
    df = df.copy()
    df["children"] = df["children"].fillna(0)
    df["country"] = df["country"].fillna("Unknown")
    df["has_agent"] = df["agent"].notna().astype(int)
    df["has_company"] = df["company"].notna().astype(int)
    df["agent"] = df["agent"].fillna(-1).astype(int).astype(str)
    df["company"] = df["company"].fillna(-1).astype(int).astype(str)

    df["arrival_month"] = df["arrival_date_month"].map(MONTHS)
    df["arrival_date"] = pd.to_datetime(dict(
        year=df["arrival_date_year"], month=df["arrival_month"],
        day=df["arrival_date_day_of_month"]))
    df["booking_date"] = df["arrival_date"] - pd.to_timedelta(df["lead_time"], unit="D")
    df["arrival_dow"] = df["arrival_date"].dt.dayofweek
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]

    # Drop rows that cannot be real stays (no guests) and negative ADR.
    df = df[(df["total_guests"] > 0) & (df["adr"] >= 0)]
    return df.reset_index(drop=True)


def booking_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return only columns knowable at booking time (no leakage)."""
    return df[CATEGORICAL + NUMERIC].copy()


def temporal_split(df: pd.DataFrame, cutoff: str = "2017-03-01"):
    """Train on bookings with arrival before `cutoff`, test on the rest.

    A random split would let the model see the future; a time-based split
    mimics deployment, where we score new bookings with a model trained on the past.
    """
    cutoff = pd.Timestamp(cutoff)
    train = df[df["arrival_date"] < cutoff]
    test = df[df["arrival_date"] >= cutoff]
    return train, test

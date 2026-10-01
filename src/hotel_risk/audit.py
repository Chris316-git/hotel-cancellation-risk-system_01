"""Quick data/leakage audit: python -m hotel_risk.audit data/raw/hotel_bookings.csv"""
import sys

from .data import LEAKAGE_COLUMNS, TARGET, booking_time_features, clean, load_raw, temporal_split


def main(path: str) -> None:
    raw = load_raw(path)
    df = clean(raw)
    X = booking_time_features(df)
    train, test = temporal_split(df)
    print(f"raw rows: {len(raw):,} | after cleaning: {len(df):,}")
    print(f"cancel rate overall: {df[TARGET].mean():.1%}")
    print(f"train: {len(train):,} ({train.arrival_date.min().date()} to {train.arrival_date.max().date()}, cancel {train[TARGET].mean():.1%})")
    print(f"test : {len(test):,} ({test.arrival_date.min().date()} to {test.arrival_date.max().date()}, cancel {test[TARGET].mean():.1%})")
    print(f"booking-time features: {X.shape[1]}")
    print("excluded as leakage:")
    for c, why in LEAKAGE_COLUMNS.items():
        print(f"  - {c}: {why}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw/hotel_bookings.csv")

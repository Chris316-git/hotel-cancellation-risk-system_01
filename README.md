# Hotel Cancellation Risk System

Predict which hotel bookings will cancel **at the moment they are made**, then turn the
risk score into a business decision (overbooking / deposit policy). Built on the public
[Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) dataset.

## Design principles
- **Booking-time features only.** Columns such as `reservation_status`, `assigned_room_type`
  and `booking_changes` are only known after the outcome and are excluded (see `LEAKAGE_COLUMNS`).
- **Temporal validation.** Train on earlier arrivals, test on later ones, as in deployment.
- **Business-impact layer** (planned): translate predicted risk into expected revenue impact.

## Roadmap
- [x] Phase 1: scaffold, data cleaning, leakage audit, temporal split
- [ ] Phase 2: baseline + gradient boosting models, temporal evaluation
- [ ] Phase 3: calibration and threshold selection
- [ ] Phase 4: business-impact simulation
- [ ] Phase 5: packaging / API / monitoring

## Quickstart
```bash
pip install -r requirements.txt && pip install -e .
# place hotel_bookings.csv in data/raw/
python -m hotel_risk.audit
pytest
```

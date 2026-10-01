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
- [x] Phase 2: baseline + gradient boosting models, temporal evaluation
- [x] Phase 3: calibration and threshold selection
- [x] Phase 4: business-impact simulation
- [ ] Phase 5: packaging / API / monitoring

## Quickstart
```bash
pip install -r requirements.txt && pip install -e .
# place hotel_bookings.csv in data/raw/
python -m hotel_risk.audit
pytest
```

## Phase 2 results (booking-time features only)

| Split | Model | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|---|
| Temporal (train < 2017-03, test after) | Logistic baseline | 0.837 | 0.790 | 0.161 |
| Temporal | HistGradientBoosting | **0.854** | **0.809** | **0.155** |
| Random (for comparison) | Logistic baseline | 0.875 | 0.828 | 0.135 |
| Random | HistGradientBoosting | 0.929 | 0.901 | 0.102 |

A random split overstates performance (GBM AUC 0.929 vs 0.854 on a temporal split), so
all reported numbers use the temporal split. Reproduce with `python -m hotel_risk.evaluate`.

## Phase 3: calibration and thresholds

Chronological train (< 2016-12) / calibration (2016-12 to 2017-02) / test (2017-03+) slices.
Calibrators and the decision threshold are fit on the calibration slice only.

| Test scores | Brier | Log loss | ECE |
|---|---|---|---|
| Raw GBM | 0.158 | 0.474 | 0.064 |
| Platt (sigmoid) | 0.155 | 0.456 | 0.049 |
| Isotonic | 0.155 | 0.456 | 0.043 |

Calibration improves probability quality without hurting ranking (AUC ~0.85). The threshold
chosen for >=80% precision on the calibration slice (0.45) delivered 73% precision / 61% recall
on test, because the cancel rate drifted from 34% to 40%. The highest-risk 10% of bookings were
100% cancellations on test (largely non-refundable-deposit bookings, a known quirk of this
dataset), so the score is most informative in the middle of the range. Phase 4 prices these
decisions in revenue terms. Reproduce: `python -m hotel_risk.calibrate`.

## Phase 4: business impact

Policy: oversell the room of a booking when its calibrated cancel probability `p` satisfies
`p * gain > (1 - p) * cost`, where gain = resell-fill-rate x ADR x nights (room re-filled) and
cost = relocation cost if the guest shows up. **Assumptions are illustrative** (fill rate 50%,
walk cost 2 nights ADR + $50) and editable in `Assumptions`; sensitivity is in `reports/phase4_impact.json`.

| Policy (test, 32,778 bookings) | Net value |
|---|---|
| No overbooking | $0 |
| Flag p >= 0.5 | $675k |
| Flag p >= 0.8 | $754k |
| **Cost-aware model** | **$1.22M** |
| Oracle (perfect foresight, upper bound) | $3.25M |

The cost-aware policy captures about 37% of the oracle upper bound and beats fixed thresholds
because it adapts to each booking's price and stay length. Value ranges from $0.4M to $3.4M
across fill-rate/walk-cost scenarios, so conclusions depend on real operating costs.
Limitations: ignores hotel capacity and demand limits and treats bookings independently.
Reproduce: `python -m hotel_risk.impact`.

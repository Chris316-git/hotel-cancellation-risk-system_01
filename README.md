# Hotel Cancellation Risk System

Roughly 4 in 10 hotel bookings in this dataset never turn into a stay. This project tries to guess
which ones, **at the moment the booking is made**, and then asks the question the hotel actually
cares about: is it worth overselling that room?

It covers the whole trip: honest data cleaning, a model, a model that is also *calibrated* (so "70%"
actually means about 70%), a dollars-and-cents decision layer, and a small API you can poke at.

## The short version

- Predicts cancellations using only what the hotel knows when the booking comes in. No peeking at
  how the stay ended.
- Tests on bookings that arrive *after* the training period, the way a real deployment would.
  A random split flatters the model (AUC 0.929 vs 0.852). We report the less flattering number.
- Turns probabilities into an overbooking policy worth about **$1.22M** on the test period
  under illustrative cost assumptions, roughly 38% of what a perfect fortune teller would earn.

## Data

The project uses the public **Hotel Booking Demand** dataset: 119,390 bookings from two hotels
(a city hotel and a resort hotel), arriving between July 2015 and August 2017. It has 32 columns,
including the `is_canceled` label. The file lives in `data/raw/hotel_bookings.csv`, so you can run
everything without hunting for data.

Reference: N. Antonio, A. de Almeida, L. Nunes, "Hotel booking demand datasets", *Data in Brief*, 22, 2019
([ISCTE repository](https://repositorio.iscte-iul.pt/handle/10071/16929)). Check the original
publication for terms of use before reusing or redistributing the data.

**Why this one**

- One row per reservation, with a real cancellation outcome. That is exactly what a cancellation
  model needs, and aggregate hotel stats can't give you.
- Arrival dates and lead times make a proper time-based split possible, which this project leans on.
- It comes with a peer-reviewed description, so you can check where it came from.
- It also includes columns that are only known *after* the outcome (`reservation_status`,
  `assigned_room_type`, `booking_changes`...), which makes it a good playground for catching leakage.

**What it is not**

- It is not recent and it is not US data: two hotels, 2015-2017. Results show the method works,
  not what your hotel's numbers will look like. Booking-level US data with cancellations is
  mostly proprietary, and I couldn't find a verifiable recent public alternative.
- Some copies of this dataset online come with relabelled hotel names, shifted dates or extra columns.
  I skipped those: the origin is unclear, and altered dates would break the time-based split.
- The money numbers in Phase 4 come from assumptions I made up and labelled as such, not from these hotels.

## How it avoids cheating

- **Booking-time features only.** Columns like `reservation_status`, `assigned_room_type` and
  `booking_changes` reveal the outcome or only exist after it, so they are excluded
  (see `LEAKAGE_COLUMNS` in `src/hotel_risk/data.py`, each with the reason).
- **Temporal validation.** Train on the past, test on the future.

## Run it yourself

```bash
python -m venv .venv && source .venv/bin/activate     # or use conda
pip install -r requirements.txt && pip install -e .

pytest -q                                   # tests (the API test runs if FastAPI is installed)
python -m hotel_risk.audit                  # data checks + the leakage column list
python -m hotel_risk.evaluate               # Phase 2: models, temporal vs random split
python -m hotel_risk.calibrate              # Phase 3: calibration + thresholds
python -m hotel_risk.impact                 # Phase 4: overbooking policy value

python -m hotel_risk.train data/raw/hotel_bookings.csv models/model.joblib
uvicorn hotel_risk.api:app --port 8000      # then open http://localhost:8000/docs
```

Your numbers may differ in the third decimal depending on library versions.

## Results

### Models (temporal split: train before 2017-03, test after)

| Split | Model | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|---|
| Temporal | Logistic baseline | 0.836 | 0.789 | 0.161 |
| Temporal | HistGradientBoosting | **0.852** | **0.807** | **0.156** |
| Random (for comparison) | Logistic baseline | 0.875 | 0.828 | 0.135 |
| Random | HistGradientBoosting | 0.929 | 0.900 | 0.102 |

The random split looks better because it lets the model learn from bookings that happen later than
the ones it is tested on. Real hotels don't get to do that, so the temporal numbers are the ones to trust.

### Calibration and thresholds

Data is split chronologically into train (before 2016-12), calibration (2016-12 to 2017-02) and
test (2017-03 onward). Calibrators and the decision threshold are fit on the calibration slice only.

| Test scores | Brier | Log loss | ECE |
|---|---|---|---|
| Raw GBM | 0.158 | 0.476 | 0.066 |
| Platt (sigmoid) | 0.155 | 0.457 | 0.047 |
| Isotonic | 0.156 | 0.460 | 0.042 |

Calibration makes the probabilities more honest without hurting ranking (AUC stays about 0.85).

Two things worth knowing:

- The threshold picked for 80% precision on the calibration slice (0.46) gave 74% precision and
  60% recall on test. The cancel rate drifted from 34% to 40% between the two periods, so a
  threshold tuned on the past doesn't transfer perfectly. Real data does this.
- The riskiest 10% of bookings were 100% cancellations on test, largely non-refundable-deposit
  bookings, a well-known quirk of this dataset. Don't read that as superhuman skill. The model
  is most useful in the messy middle.

### Business impact

The policy: oversell a booking's room when `p * gain > (1 - p) * cost`, where `p` is the calibrated
cancel probability, *gain* is what you earn if the room is re-sold (resell rate x ADR x nights), and
*cost* is what it takes to relocate a guest who shows up anyway.

The assumptions are made up on purpose and easy to change in `Assumptions`: 50% chance a freed room
is re-sold, relocation costs 2 nights of ADR plus $50.

| Policy (test period, 32,778 bookings) | Net value |
|---|---|
| No overbooking | $0 |
| Flag p >= 0.5 | $774k |
| Flag p >= 0.8 | $679k |
| **Cost-aware model** | **$1.22M** |
| Perfect foresight (upper bound) | $3.25M |

The cost-aware policy beats fixed thresholds because it accounts for each booking's price and length of
stay. Across the scenarios in `reports/phase4_impact.json` the value ranges from $0.4M to $3.4M,
so the real answer depends on real operating costs. It also ignores hotel capacity and demand limits
and treats bookings as independent, so treat the dollars as an illustration, not a forecast.

## The API

`POST /score` takes a list of booking-time records and returns, for each, a calibrated cancel
probability, a risk band (low / medium / high) and whether overselling the room is recommended
under the Phase 4 assumptions. Records with no guests or a negative ADR are rejected. `GET /health`
says whether it's alive.

A Dockerfile is included, and GitHub Actions runs the tests on every push
(`.github/workflows/ci.yml`). If this ran in production, the first thing to watch would be cancel-rate
drift (see the 34% to 40% jump above), with scheduled retraining.

```bash
docker build -t hotel-risk . && docker run -p 8000:8000 -v $PWD/models:/app/models hotel-risk
```

## Roadmap

- [x] Phase 1: scaffold, data cleaning, leakage audit, temporal split
- [x] Phase 2: baseline and gradient boosting, temporal evaluation
- [x] Phase 3: calibration and threshold selection
- [x] Phase 4: business-impact simulation
- [x] Phase 5: model packaging, API, tests, CI

## License

The code is under the MIT License (see `LICENSE`). That covers the code only. The dataset in
`data/raw/` is subject to the terms of its original publication (see the Data section).

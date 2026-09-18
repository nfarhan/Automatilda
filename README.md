# NYC Taxi Fare Prediction

A regression model that estimates NYC taxi fares before a ride begins, built as a
portfolio project simulating a real consulting engagement between **Automatidata**
and the **NYC Taxi & Limousine Commission (TLC)**.

> This project was completed as part of a Coursera capstone. The client scenario,
> team names, and business context are fictional; the dataset is real 2017 TLC
> trip data.

## Business problem

TLC wants riders and drivers to see a fare estimate before a trip starts. That
means the model has to be reliable not just on average, but across the realistic
range of trip types a rider might take short crosstown hops, standard metered
rides, and fixed-rate airport runs, not just on whichever slice makes the
headline metric look best.

## Result

**Random Forest**, selected over XGBoost and Linear Regression after model
comparison and segment-level error analysis (not on aggregate RMSE alone. See
[Why Random Forest, not XGBoost](#why-random-forest-not-xgboost) below).

| Model | RMSE | MAE | R² | Verdict |
|---|---|---|---|---|
| **Random Forest** | $2.154 | **$0.343** | 0.964 | **Recommended**: most consistent across trip types |
| XGBoost | $2.076 | $0.354 | 0.966 | Better on short/cheap trips only; less stable on rare fare types |
| Linear Regression | $3.848 | $0.913 | 0.884 | Not viable: misses airport flat rate by $8+ on average |

On a typical trip, the recommended model's fare estimate is within **34 cents**
of the actual metered fare.

## Data

- **Source:** [2017 Yellow Taxi Trip Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), NYC Open Data / TLC
- **Sample:** 1,000,000 trips (4-day window, Aug 11–15, 2017), drawn from a
  113M-row source table for tractable runtimes
- **Final feature matrix:** 991,998 rows × 18 features + target, after cleaning
  removed 0.80% of rows (negative fares, zero-distance trips, invalid rate codes,
  near-duplicate records)

## Pipeline

The project follows a phased, notebook-per-phase structure, with each phase
importing shared config, constants, and utilities from `src/config.py`
(paths, `RANDOM_STATE`, `TARGET_COL`, DuckDB connection, `quick_sql()`,
`save_figure()`, logger).

| Phase | What it does | Key output |
|---|---|---|
| **0. Kickoff** | Environment setup, config module, data load, DuckDB init | Reusable `src/config.py` |
| **1. Data Audit** | Schema, missingness, duplicate, and range-contract validation | Formal data contract (0 missing, 518 near-dupes, 9 range violations) |
| **2. EDA** | Target distribution, correlations, temporal patterns, fare drivers, outlier profiling | Identified `total_amount`/`tip_amount` as leakage; confirmed JFK flat-rate step-change |
| **3. Cleaning** | Enforced Phase 1 contracts, dropped leakage/near-constant columns | `taxi_clean.parquet`: 991,998 × 12 |
| **4. Feature Engineering** | Temporal features, trip duration, log1p transforms, one-hot encoding, binning | `taxi_features.parquet`: 991,998 × 19 |
| **5. Model Building** | 80/20 split, Linear Regression → Random Forest → XGBoost, `TransformedTargetRegressor` for log-scale fitting | Three fitted models, initial comparison table |
| **6. Model Evaluation** | Residual analysis, error by fare range/trip type, feature importance, final recommendation | **Random Forest selected**; executive summary |

## Why Random Forest, not XGBoost

Phase 5's aggregate metrics gave XGBoost the edge on RMSE and R². Phase 6 broke
that number down by fare range, rate code, and trip duration, and the picture
changed:

- XGBoost's RMSE advantage was concentrated almost entirely in the **$0–5 fare /
  0–5 minute** bucket (13% of trips). Outside that segment, Random Forest matched
  or beat it.
- On the **Standard rate code** (97% of all trips) and the **$10–30 fare range**
  (63% of trips, the bulk of realistic rides), Random Forest won on both RMSE
  and MAE.
- On **MAE**, the more interpretable "typical error per trip" metric, Random
  Forest won almost every segment tested.
- On **rare rate codes** (Nassau/Westchester, Newark), XGBoost's accuracy dropped
  sharply relative to Random Forest, likely gradient boosting overfitting to
  sparse categories, a pattern bagging is more robust to.

Linear Regression was disqualified outright: it misses the JFK flat $52 fare by
an average of $8 because it can't learn a fixed rate, it only knows how to scale
predictions with distance.

**Caveat:** this isn't a landslide. The margins are small almost everywhere
except JFK/Nassau-Westchester error, and neither tree model was hyperparameter-
tuned. A tuned XGBoost could plausibly close the gap on rare rate codes. This
recommendation reflects the models as built, evaluated against realistic
deployment segments rather than a single aggregate score.

## What drives the prediction

`log1p(trip_distance)` and `log1p(trip_duration_min)` together account for
~97% of feature importance in both tree models, the fare is overwhelmingly a
function of how far and how long the trip is, which is exactly what should be
true for a metered fare. Rate-code flags (JFK, Newark, Nassau/Westchester,
Negotiated) capture the flat-rate step-changes on top of that. Time-of-day
features (rush hour, weekend, hour of day) added negligible model value once
distance and duration were known, despite looking meaningful in raw EDA
averages, they turned out to be a proxy the two continuous features already
captured.

## Tech stack

- **Python:** pandas, NumPy, scikit-learn, XGBoost
- **DuckDB:** SQL-layer validation and cross-checks throughout
- **Parquet:** (snappy compression), standard serialization between phases
- **Jupyter:** one notebook per phase

## Limitations & next steps

- **Sample window:** trained on a 4-day August 2017 slice; a production model
  should retrain on a full-year sample to capture seasonal effects
- **No hyperparameter tuning:** both tree models ran with reasonable defaults;
  tuning is the most likely lever to close the Random Forest/XGBoost gap
- **Unused location data:** `pulocationid`/`dolocationid` were available but
  dropped in Phase 4; zone-level features (e.g., Manhattan-to-Manhattan vs.
  cross-borough) are a promising next iteration

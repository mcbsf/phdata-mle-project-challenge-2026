# PR 2: ZIP-scoped KNN imputation

## Intent

PR 1’s incomplete-request path searched all 21,613 historical donors even though each request already supplies a required zipcode. This change restricts neighbors to the request’s zipcode to reduce work during prediction.

## What changed

`src/api/endpoints.py` now fits 70 ZIP-specific `KNNImputer`s at startup, one per supported zipcode, and reuses the shared feature scaler. An incomplete request selects its zipcode’s prefitted imputer; no fitting happens per request. Complete requests still bypass scaling and imputation and use the existing model input unchanged. `test/unit/test_api_unit.py` verifies that all supported zipcodes have an imputer.

## Validation and latency

The Docker unit suite passed 21 tests; the imputer map covers all 70 supported zipcodes, and all 100 future examples returned finite predictions. On the same benchmark harness as PR 1, the two-field-missing request measured:

| Load | PR 1 global KNN p50 / p95 | PR 2 ZIP KNN p50 / p95 |
| --- | ---: | ---: |
| Sequential | 90.760 / 127.619 ms | 7.542 / 8.315 ms |
| 4 concurrent workers × 80 requests | 281.022 / 484.527 ms | 28.497 / 30.333 ms |

These are results from one local run. Startup resource loading increased from 69.68 ms to 85.60 ms (+15.92 ms), excluding imports, because the ZIP-specific imputers are fitted before serving requests.

## Held-out imputation comparison

This accuracy check uses seven property fields plus zipcode demographics in 33 features. A ZIP-stratified 80/20 split (`random_state=42`) produced 17,290 training rows and 4,323 holdout rows. From the holdout, 1,000 rows (`sample(random_state=42)`) covered all 70 ZIPs. A `default_rng(42)` masked exactly 150 values per property field (1,050 total); both methods used the same rows and mask. The scaler was fitted on training rows only. Global KNN used all training donors; ZIP KNN used only training donors from the query ZIP, so no holdout row entered fitting.

For each field, RMSE and MAE on masked values were divided by that field’s training-only population standard deviation (`ddof=0`), then macro-averaged equally across the seven fields. Results are mixed: normalized macro RMSE was 0.5392 for global KNN and 0.5442 for ZIP KNN; normalized macro MAE was 0.3083 and 0.3065, respectively. ZIP KNN had lower RMSE on 4/7 fields; global KNN had lower RMSE on 3/7. Accuracy did not improve uniformly.

## Limitation

When all seven home fields are absent, rows for the same zipcode have identical known demographic features. The ZIP-specific KNN’s five neighbors therefore tie on distance; the imputation has no property-similarity signal and relies on zipcode-level context. This is especially relevant for interpreting the all-fields-missing estimate.

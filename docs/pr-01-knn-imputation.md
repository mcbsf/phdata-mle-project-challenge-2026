# PR 1: KNN imputation for incomplete home requests

## Intent

Accept incomplete `/predict` requests while requiring `zipcode`. Estimate missing home features from historical sales and zipcode demographics; preserve the existing path for complete requests.

## What changed

- `src/api/endpoints.py` makes the seven home fields optional, uses KNN for incomplete requests, and returns HTTP 422 for unknown zipcodes.
- `src/main.py` loads the model, donor data, scaler, and imputer once at application startup.
- `test/conftest.py` starts FastAPI’s lifespan for tests. `test/unit/test_api_unit.py` covers health, complete and missing input, and zipcode validation.
- `README.md` documents missing-field behavior and Docker unit tests. `requirements.txt` is unchanged; scikit-learn was already installed.

## How imputation works

At startup, historical sales are joined to zipcode demographics. The 33 model-ordered features form the donor matrix; sale price and raw zipcode are excluded. `StandardScaler` keeps large-unit values such as square footage from dominating distances. For each missing value, `KNNImputer` uses up to five nearest donors, weighted toward closer rows.

For incomplete requests, demographics for the required zipcode are added, the row is scaled, imputed, then returned to the model’s feature scale. Complete requests skip scaling and imputation and go straight to the model with their original feature values. The standard complete sample remains `253980.0`.

## Request examples

Null and omitted home fields are accepted:

```json
{"zipcode":"98042","bedrooms":3,"bathrooms":null,"sqft_living":1500,"sqft_lot":5000,"floors":1,"sqft_above":1200}
```

All seven home fields may be missing:

```json
{"zipcode":"98042"}
```

Missing, null, or unknown zipcode returns HTTP 422.

## Docker validation

```bash
docker build -t mle-project-challenge-2026 .
docker run --rm -d -p 8000:8000 --name housing-api mle-project-challenge-2026
curl -i http://127.0.0.1:8000/health
curl -sS -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d '{"bedrooms":3,"bathrooms":2,"sqft_living":1500,"sqft_lot":5000,"floors":1,"sqft_above":1200,"sqft_basement":300,"zipcode":"98042"}'
curl -sS -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d '{"zipcode":"98042","bedrooms":3,"bathrooms":null,"sqft_living":1500,"sqft_lot":5000,"floors":1,"sqft_above":1200}'
curl -sS -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d '{"zipcode":"98042"}'
curl -i -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d '{"zipcode":"99999"}'
curl -i -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d '{"bedrooms":3}'
docker stop housing-api

docker build -f Dockerfile.test -t mle-api-test .
docker run --rm mle-api-test pytest test/unit -v
```

## Validation and timing

The expanded Docker unit suite passed 20/20 tests, covering every field as null or omitted and zipcode as null or omitted. Missing, null, or unknown zipcode returned 422. The complete sample returned `253980.0`; multiple and all-seven-missing requests returned finite predictions. All 100 future examples were finite. The production image built and `/health` returned 200. The 33-feature donor matrix contains 21,613 rows and no all-NaN columns.

Same-host `urllib` timings in milliseconds (p50 / p95 / mean; no warm-up, startup excluded):

| Request/load | Baseline | Current |
| --- | ---: | ---: |
| Complete, 30 sequential | 29.199 / 57.039 / 49.111 | 4.916 / 5.940 / 7.957 |
| Complete, 4 concurrent workers × 80 requests | 118.887 / 150.909 / 157.361 | 18.569 / 21.807 / 18.612 |
| Current two-field-missing payload, sequential | Rejected | 90.760 / 127.619 / 83.485 |
| Current two-field-missing payload, 4 concurrent workers × 80 requests | Rejected | 281.022 / 484.527 / 293.929 |

The old API rejected incomplete input, so there is no missing-path baseline. Its current timings motivate a narrow follow-up performance PR, measured against this version. This is one same-host run, not a production load test.

## Limits and trade-offs

Imputed values are estimates. With all seven home fields missing, the estimate relies on zipcode demographics and historical donors and lacks property-specific detail. Unsupported zipcodes fail; imputation uncertainty is not reported.

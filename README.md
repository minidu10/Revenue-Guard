# Customer Churn Prediction Service

## Problem
Companies lose money when customers leave. This project predicts which
customers are likely to leave, so the business can act early.

Full write-up: [docs/problem-definition.md](docs/problem-definition.md)

## What I am building
An end-to-end machine learning service:
- data pipeline
- trained model
- REST API for predictions
- tests
- Docker container
- deployment
- basic monitoring

## Status
Stages 1-7 complete: data, problem definition, baseline model, training
pipeline, FastAPI service, automated tests, Docker containerization.
Next: cloud deployment and basic monitoring.

## Project layout

## Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train the model
```powershell
python -m src.train
```
Writes `models/churn_model.joblib` and `models/metrics.json`.

## Run the API locally
```powershell
uvicorn src.api:app --reload
```
Interactive docs: http://127.0.0.1:8000/docs

## Run in Docker
```powershell
docker build -t churn-prediction-service .
docker run -d --name churn-api -p 8000:8000 churn-prediction-service
```

## Endpoints
`GET /health`
```json
{"status": "ok"}
```

`POST /predict` — send the 19 customer attributes, get back a probability.
```json
{"churn_probability": 0.8064, "churn_prediction": 1, "threshold_used": 0.4}
```

Example request:
```powershell
$body = @{
    gender="Female"; SeniorCitizen=0; Partner="Yes"; Dependents="No"; tenure=1
    PhoneService="No"; MultipleLines="No phone service"; InternetService="DSL"
    OnlineSecurity="No"; OnlineBackup="Yes"; DeviceProtection="No"; TechSupport="No"
    StreamingTV="No"; StreamingMovies="No"; Contract="Month-to-month"
    PaperlessBilling="Yes"; PaymentMethod="Electronic check"
    MonthlyCharges=29.85; TotalCharges=29.85
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -Body $body -ContentType "application/json"
```

## Tests
```powershell
python -m pytest -q
```
7 tests: data cleaning plus API health, valid prediction, missing field (422),
wrong type (422).

## Model
Logistic regression with scaling and one-hot encoding, `class_weight="balanced"`,
in a single scikit-learn pipeline so the same preprocessing runs at training and
at prediction time.

| metric | value |
| --- | --- |
| ROC-AUC | 0.8416 |
| recall | 0.8663 |
| precision | 0.4662 |
| accuracy | 0.7012 |

Decision threshold is 0.4, not the default 0.5. Missing a leaver costs more than
a wasted retention offer, so the threshold is tuned for recall. At 0.4 the model
catches 324 of 374 real leavers at the cost of 371 false alarms. Threshold
comparison: [docs/experiments.md](docs/experiments.md)

Accuracy looks low next to the 73% always-predict-"No" baseline, but that
baseline catches zero churners, which is why it is not the metric used here.

## Real problems solved

- **Silent data quality issue**: `TotalCharges` had 11 blank values with no
  nulls reported by pandas. All 11 were customers with `tenure = 0`
  (brand new, not yet billed) — a real business rule hiding as bad data,
  not missing data. Filled with 0.0 instead of dropping the rows, since
  dropping would have removed exactly the newest, highest-risk customers.

- **Data leakage risk**: used a single scikit-learn `Pipeline` for scaling,
  encoding, and the classifier together, so preprocessing is always fit on
  training data only, both at training time and at prediction time.

- **Training/serving skew across dependency versions**: `requirements.txt`
  (full dev environment) is too heavy and version-incompatible for
  production. Built a separate `requirements-prod.txt` pinned to the exact
  scikit-learn/numpy/scipy/joblib versions the model was trained under.
  A one-version mismatch (scikit-learn 1.7.2 vs. 1.9.0) let the container
  build and start successfully, then failed only on the first real
  prediction with `AttributeError: 'LogisticRegression' object has no
  attribute 'multi_class'`. This is a dangerous class of bug because
  nothing looks wrong until a request actually arrives.

- **Windows/PowerShell tooling friction**: PowerShell aliases `curl` to
  `Invoke-WebRequest`, which does not accept standard curl syntax. Used
  `curl.exe` or native `Invoke-RestMethod` instead.

## Requirements files
- `requirements.txt` — full dev environment (Jupyter, matplotlib, pytest).
- `requirements-prod.txt` — API only, pinned to the exact versions the model
  was trained under.

Those pins matter. `joblib.load()` succeeds across a scikit-learn version gap and
then fails on the first prediction, so the container starts healthy and only
breaks when a request arrives. Building the image with scikit-learn 1.7.2 against
a model trained on 1.9.0 produced:

If the model is retrained on new library versions, regenerate `requirements-prod.txt`
and rebuild the image together.

## Data
IBM Telco Customer Churn, 7043 rows, 21 columns. `TotalCharges` is stored as text
with 11 blank values; those are coerced to numeric and filled with 0.0.
`customerID` is dropped.

## Live demo
https://revenue-guard-production.up.railway.app/docs
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def make_valid_customer() -> dict:
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    }


def test_health_check_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_valid_probability():
    response = client.post("/predict", json=make_valid_customer())
    assert response.status_code == 200

    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in (0, 1)


def test_predict_rejects_missing_field():
    bad_customer = make_valid_customer()
    del bad_customer["tenure"]

    response = client.post("/predict", json=bad_customer)
    assert response.status_code == 422


def test_predict_rejects_wrong_type():
    bad_customer = make_valid_customer()
    bad_customer["MonthlyCharges"] = "not_a_number"

    response = client.post("/predict", json=bad_customer)
    assert response.status_code == 422
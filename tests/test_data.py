import pandas as pd

from src.data import clean


def make_fake_raw_df() -> pd.DataFrame:
    return pd.DataFrame({
        "customerID": ["0001-AAA", "0002-BBB"],
        "gender": ["Female", "Male"],
        "SeniorCitizen": [0, 1],
        "Partner": ["Yes", "No"],
        "Dependents": ["No", "No"],
        "tenure": [0, 24],
        "PhoneService": ["Yes", "Yes"],
        "MultipleLines": ["No", "No"],
        "InternetService": ["DSL", "Fiber optic"],
        "OnlineSecurity": ["No", "Yes"],
        "OnlineBackup": ["No", "Yes"],
        "DeviceProtection": ["No", "Yes"],
        "TechSupport": ["No", "No"],
        "StreamingTV": ["No", "Yes"],
        "StreamingMovies": ["No", "Yes"],
        "Contract": ["Month-to-month", "Two year"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod": ["Electronic check", "Mailed check"],
        "MonthlyCharges": [29.85, 89.10],
        "TotalCharges": [" ", "2138.40"],
        "Churn": ["No", "Yes"],
    })


def test_clean_converts_blank_total_charges_to_zero():
    raw = make_fake_raw_df()
    cleaned = clean(raw)
    assert cleaned.loc[0, "TotalCharges"] == 0.0


def test_clean_converts_churn_to_binary():
    raw = make_fake_raw_df()
    cleaned = clean(raw)
    assert cleaned.loc[0, "Churn"] == 0
    assert cleaned.loc[1, "Churn"] == 1


def test_clean_drops_customer_id():
    raw = make_fake_raw_df()
    cleaned = clean(raw)
    assert "customerID" not in cleaned.columns
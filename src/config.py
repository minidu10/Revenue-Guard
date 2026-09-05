from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "churn_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

TARGET = "Churn"
DROP_COLUMNS = ["customerID"]
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

RANDOM_STATE = 42
TEST_SIZE = 0.2
DECISION_THRESHOLD = 0.4
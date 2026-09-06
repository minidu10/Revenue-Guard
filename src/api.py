import joblib
import pandas as pd
from fastapi import FastAPI

from src import config
from src.logging_config import get_logger
from src.schemas import CustomerFeatures, PredictionResponse

app = FastAPI(title="Churn Prediction Service")

model = joblib.load(config.MODEL_PATH)
logger = get_logger()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    input_df = pd.DataFrame([customer.model_dump()])
    probability = model.predict_proba(input_df)[0, 1]
    prediction = int(probability >= config.DECISION_THRESHOLD)

    logger.info(
        "prediction probability=%.4f prediction=%d tenure=%d MonthlyCharges=%.2f Contract=%s",
        probability,
        prediction,
        customer.tenure,
        customer.MonthlyCharges,
        customer.Contract,
    )

    return PredictionResponse(
        churn_probability=round(float(probability), 4),
        churn_prediction=prediction,
        threshold_used=config.DECISION_THRESHOLD,
    )
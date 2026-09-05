import json

import joblib
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src import config
from src.data import clean, load_raw, split_features_target
from src.pipeline import build_pipeline


def evaluate(model, X_test, y_test) -> dict:
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= config.DECISION_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "threshold": config.DECISION_THRESHOLD,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main() -> None:
    df = clean(load_raw())
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    model = build_pipeline(list(X.columns))
    model.fit(X_train, y_train)

    metrics = evaluate(model, X_test, y_test)
    metrics["n_train"] = int(len(X_train))
    metrics["n_test"] = int(len(X_test))

    config.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))
    print(f"Model saved to: {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
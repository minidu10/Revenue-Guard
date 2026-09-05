from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def build_pipeline(feature_columns: list) -> Pipeline:
    numeric = [c for c in config.NUMERIC_FEATURES if c in feature_columns]
    categorical = [c for c in feature_columns if c not in numeric]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )

    return Pipeline(steps=[
        ("preprocess", preprocess),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
        )),
    ])
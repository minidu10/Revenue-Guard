import pandas as pd

from src import config


def load_raw() -> pd.DataFrame:
    return pd.read_csv(config.RAW_DATA_PATH)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    df = df.drop(columns=config.DROP_COLUMNS)
    df[config.TARGET] = (df[config.TARGET] == "Yes").astype(int)
    return df


def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=[config.TARGET])
    y = df[config.TARGET]
    return X, y
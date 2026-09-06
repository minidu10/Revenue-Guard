import logging
from pathlib import Path

from src import config

LOG_PATH = config.MODEL_DIR / "predictions.log"


def get_logger() -> logging.Logger:
    logger = logging.getLogger("churn_api")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(LOG_PATH)
    formatter = logging.Formatter("%(asctime)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
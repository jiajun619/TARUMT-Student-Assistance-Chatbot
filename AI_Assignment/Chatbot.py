import random
from pathlib import Path

import joblib
import numpy as np

from NLP import rejoin
from Responses import get_response


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

_logistic_pipeline = joblib.load(MODEL_DIR / "logistic_pipeline.joblib")
_linearsvc_pipeline = joblib.load(MODEL_DIR / "linearsvc_pipeline.joblib")

def get_selected_model() -> str:
    selected_model_path = (
        MODEL_DIR / "selected_model.txt"
    )

    if selected_model_path.exists():
        model_name = (
            selected_model_path
            .read_text(encoding="utf-8")
            .strip()
        )

        if model_name in {
            "Logistic Regression",
            "LinearSVC",
        }:
            return model_name

    # Select LinearSVC by default since it is a better overall compared to Logistic Regression
    return "LinearSVC"


def predict_intent(message: str, model_name: str | None = None) -> tuple[str, float]:

    # Apply the same NLP preprocessing used in training
    cleaned = rejoin(message)

    if not cleaned:
        return "unknown", 0.0

    # Use the best model selected by TrainModels.py if it is not manually selected
    if model_name is None:
        model_name = get_selected_model()

    if model_name == "Logistic Regression":
        probabilities = _logistic_pipeline.predict_proba([cleaned])[0]
        best_index = int(np.argmax(probabilities))
        intent = _logistic_pipeline.classes_[best_index]
        score = float(probabilities[best_index])

    else:
        scores = _linearsvc_pipeline.decision_function([cleaned])[0]
        best_index = int(np.argmax(scores))
        intent = _linearsvc_pipeline.classes_[best_index]
        score = float(scores[best_index])

    return str(intent), score


def chatbot(message: str, model_name: str | None = None) -> tuple[str, str, float]:
    intent, score = predict_intent(message, model_name)

    response = get_response(message, intent)

    return response, intent, score
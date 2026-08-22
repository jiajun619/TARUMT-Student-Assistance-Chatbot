import random
from pathlib import Path

import joblib
import numpy as np

from NLP import rejoin
from Responses import responses_data


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

_vectorizer = joblib.load(MODEL_DIR / "vectorizer.joblib")
_logistic_model = joblib.load(MODEL_DIR / "logistic_model.joblib")
_linearsvc_model = joblib.load(MODEL_DIR / "linearsvc_model.joblib")


def predict_intent(message: str, model_name: str = "LinearSVC") -> tuple[str, float]:
    cleaned = rejoin(message)

    if not cleaned:
        return "unknown", 0.0

    X = _vectorizer.transform([cleaned])

    if model_name == "Logistic Regression":
        probabilities = _logistic_model.predict_proba(X)[0]
        best_index = int(np.argmax(probabilities))
        intent = _logistic_model.classes_[best_index]
        score = float(probabilities[best_index])

    else:
        scores = _linearsvc_model.decision_function(X)[0]
        best_index = int(np.argmax(scores))
        intent = _linearsvc_model.classes_[best_index]
        score = float(scores[best_index])

    return intent, score


def chatbot(message: str, model_name: str = "LinearSVC") -> tuple[str, str, float]:
    intent, score = predict_intent(message, model_name)

    response = random.choice(
        responses_data.get(intent, responses_data["unknown"])
    )

    return response, intent, score
from NLP import rejoin
from Responses import responses_data
import random
import joblib

_vectorizer = joblib.load("model/vectorizer.joblib")
_model = joblib.load("model/model.joblib")

CONFIDENCE_THRESHOLD = 0.35


def predict_intent(message: str):
    cleaned = rejoin(message)
    X = _vectorizer.transform([cleaned])
    probs = _model.predict_proba(X)[0]
    best_idx = probs.argmax()
    intent = _model.classes_[best_idx]
    confidence = probs[best_idx]
    if confidence < CONFIDENCE_THRESHOLD:
        return "unknown", confidence
    return intent, confidence


def chatbot(message: str) -> str:
    intent, confidence = predict_intent(message)
    return random.choice(responses_data.get(intent, responses_data["unknown"]))
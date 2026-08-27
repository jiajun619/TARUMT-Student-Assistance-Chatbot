from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)

from NLP import rejoin

BASE_DIR = Path(__file__).resolve().parent
UNSEEN_DATASET_PATH = (
    BASE_DIR
    / "datasets"
    / "unseen_test.csv"
)

MODEL_DIR = BASE_DIR / "model"


# 1. Load unseen test dataset
test_data = pd.read_csv(UNSEEN_DATASET_PATH)

print("Unseen test dataset loaded.")
print("Dataset shape:", test_data.shape)
print()


# 2. Preprocess questions
test_data["processed_question"] = test_data["question"].apply(rejoin)

X_test = test_data["processed_question"]
y_test = test_data["expected_intent"]


# 3. Load complete pipelines
logistic_pipeline = joblib.load(MODEL_DIR / "logistic_pipeline.joblib")

linearsvc_pipeline = joblib.load(MODEL_DIR / "linearsvc_pipeline.joblib")


# 4. Evaluate Logistic Regression
logistic_predictions = logistic_pipeline.predict(X_test)
logistic_accuracy = accuracy_score(y_test, logistic_predictions)
logistic_macro_f1 = f1_score(y_test, logistic_predictions, average="macro", zero_division=0)

print("=" * 60)
print("LOGISTIC REGRESSION - UNSEEN TEST RESULTS")
print("=" * 60)

print(f"Accuracy: {logistic_accuracy:.4f}")

print(f"Macro F1: {logistic_macro_f1:.4f}")

print()

print(
    classification_report(
        y_test,
        logistic_predictions,
        zero_division=0,
    )
)

logistic_results = pd.DataFrame({
    "question": test_data["question"],
    "expected_intent": y_test,
    "predicted_intent": logistic_predictions,
    "correct": y_test == logistic_predictions,
})

print("Detailed predictions:")
print(logistic_results.to_string(index=False))
print()


# 5. Evaluate LinearSVC
linearsvc_predictions = linearsvc_pipeline.predict(X_test)
linearsvc_accuracy = accuracy_score(y_test, linearsvc_predictions)
linearsvc_macro_f1 = f1_score(y_test, linearsvc_predictions, average="macro", zero_division=0)

print("=" * 60)
print("LINEARSVC - UNSEEN TEST RESULTS")
print("=" * 60)

print(f"Accuracy: {linearsvc_accuracy:.4f}")

print(f"Macro F1: {linearsvc_macro_f1:.4f}")

print()

print(
    classification_report(
        y_test,
        linearsvc_predictions,
        zero_division=0,
    )
)

linearsvc_results = pd.DataFrame({
    "question": test_data["question"],
    "expected_intent": y_test,
    "predicted_intent": linearsvc_predictions,
    "correct": y_test == linearsvc_predictions,
})

print("Detailed predictions:")
print(linearsvc_results.to_string(index=False))
print()


# 6. Final comparison
print("=" * 60)
print("UNSEEN TEST MODEL COMPARISON")
print("=" * 60)

print(f"Logistic Regression Accuracy: {logistic_accuracy:.4f}")
print(f"Logistic Regression Macro F1: {logistic_macro_f1:.4f}")

print(f"LinearSVC Accuracy: {linearsvc_accuracy:.4f}")
print(f"LinearSVC Macro F1: {linearsvc_macro_f1:.4f}")
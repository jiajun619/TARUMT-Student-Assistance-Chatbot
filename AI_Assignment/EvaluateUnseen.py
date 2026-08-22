import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
)

from NLP import rejoin


# 1. Load unseen test dataset
test_data = pd.read_csv("datasets/unseen_test.csv")

print("Unseen test dataset loaded.")
print("Dataset shape:", test_data.shape)
print()


# 2. Preprocess questions
test_data["processed_question"] = test_data["question"].apply(rejoin)

X_test = test_data["processed_question"]
y_test = test_data["expected_intent"]


# 3. Load vectorizer and models
vectorizer = joblib.load("model/vectorizer.joblib")
logistic_model = joblib.load("model/logistic_model.joblib")
linearsvc_model = joblib.load("model/linearsvc_model.joblib")


# 4. Convert unseen questions into TF-IDF features
X_test_vectorized = vectorizer.transform(X_test)


# 5. Evaluate Logistic Regression
logistic_predictions = logistic_model.predict(X_test_vectorized)
logistic_accuracy = accuracy_score(y_test, logistic_predictions)

print("=" * 60)
print("LOGISTIC REGRESSION - UNSEEN TEST RESULTS")
print("=" * 60)
print("Accuracy:", logistic_accuracy)
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


# 6. Evaluate LinearSVC
linearsvc_predictions = linearsvc_model.predict(X_test_vectorized)
linearsvc_accuracy = accuracy_score(y_test, linearsvc_predictions)

print("=" * 60)
print("LINEARSVC - UNSEEN TEST RESULTS")
print("=" * 60)
print("Accuracy:", linearsvc_accuracy)
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


# 7. Final comparison
print("=" * 60)
print("UNSEEN TEST MODEL COMPARISON")
print("=" * 60)
print("Logistic Regression Accuracy:", logistic_accuracy)
print("LinearSVC Accuracy:", linearsvc_accuracy)
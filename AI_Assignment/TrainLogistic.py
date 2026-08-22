from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split

from NLP import rejoin


# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "datasets" / "tarumt_dataset.csv"


# 1. Read dataset
data = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully.")
print("Dataset shape:", data.shape)
print()


# 2. Apply NLP preprocessing
data["processed_question"] = data["question"].apply(rejoin)

print("Example after preprocessing:")
print(data[["question", "processed_question"]].head())
print()


# 3. Separate input and output
X = data["processed_question"]
y = data["intent"]


# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print()


# 5. Convert text into TF-IDF features
vectorizer = TfidfVectorizer()

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


# 6. Train Logistic Regression
logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
)

logistic_model.fit(
    X_train_vectorized,
    y_train,
)


# 7. Make predictions
logistic_predictions = logistic_model.predict(
    X_test_vectorized
)

logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions,
)


# 8. Display evaluation results
print("=" * 60)
print("LOGISTIC REGRESSION RESULTS")
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


# 9. Display detailed predictions
print("=" * 60)
print("LOGISTIC REGRESSION DETAILED PREDICTIONS")
print("=" * 60)

logistic_results = pd.DataFrame({
    "question": X_test,
    "actual_intent": y_test,
    "predicted_intent": logistic_predictions,
})

logistic_results["correct"] = (
    logistic_results["actual_intent"]
    == logistic_results["predicted_intent"]
)

print(logistic_results.to_string(index=False))
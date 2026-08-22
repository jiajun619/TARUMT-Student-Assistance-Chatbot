import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from NLP import rejoin


# 1. Read dataset
data = pd.read_csv("datasets/tarumt_dataset.csv")

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


# 5. Convert text into numbers using TF-IDF
vectorizer = TfidfVectorizer()

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


# 6. Train Logistic Regression
logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
)

logistic_model.fit(X_train_vectorized, y_train)

logistic_predictions = logistic_model.predict(X_test_vectorized)

logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions,
)

print("=" * 50)
print("LOGISTIC REGRESSION RESULTS")
print("=" * 50)
print("Accuracy:", logistic_accuracy)
print()
print(
    classification_report(
        y_test,
        logistic_predictions,
        zero_division=0,
    )
)

print("=" * 50)
print("LOGISTIC REGRESSION DETAILED PREDICTIONS")
print("=" * 50)

logistic_results = pd.DataFrame({
    "question": X_test,
    "actual_intent": y_test,
    "predicted_intent": logistic_predictions,
})

print(logistic_results.to_string(index=False))
print()

# 7. Train LinearSVC
linearsvc_model = LinearSVC(
    random_state=42,
)

linearsvc_model.fit(X_train_vectorized, y_train)

linearsvc_predictions = linearsvc_model.predict(X_test_vectorized)

linearsvc_accuracy = accuracy_score(
    y_test,
    linearsvc_predictions,
)

print("=" * 50)
print("LINEARSVC RESULTS")
print("=" * 50)
print("Accuracy:", linearsvc_accuracy)
print()
print(
    classification_report(
        y_test,
        linearsvc_predictions,
        zero_division=0,
    )
)

print("=" * 50)
print("LINEARSVC DETAILED PREDICTIONS")
print("=" * 50)

linearsvc_results = pd.DataFrame({
    "question": X_test,
    "actual_intent": y_test,
    "predicted_intent": linearsvc_predictions,
})

print(linearsvc_results.to_string(index=False))
print()


# 8. Compare models
print("=" * 50)
print("MODEL COMPARISON")
print("=" * 50)
print("Logistic Regression Accuracy:", logistic_accuracy)
print("LinearSVC Accuracy:", linearsvc_accuracy)


# 9. Retrain final models using the complete dataset

final_vectorizer = TfidfVectorizer()

X_full_vectorized = final_vectorizer.fit_transform(X)

final_logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
)

final_linearsvc_model = LinearSVC(
    random_state=42,
)

final_logistic_model.fit(X_full_vectorized, y)
final_linearsvc_model.fit(X_full_vectorized, y)


# 10. Create model folder
os.makedirs("model", exist_ok=True)


# 11. Save final vectorizer and models
joblib.dump(
    final_vectorizer,
    "model/vectorizer.joblib",
)

joblib.dump(
    final_logistic_model,
    "model/logistic_model.joblib",
)

joblib.dump(
    final_linearsvc_model,
    "model/linearsvc_model.joblib",
)

print()
print("Final models retrained using the complete dataset.")
print("Models saved successfully in the model folder.")
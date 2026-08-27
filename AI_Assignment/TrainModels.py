from __future__ import annotations
from pathlib import Path

import joblib
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.svm import LinearSVC

from NLP import rejoin


# Project paths
BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    BASE_DIR
    / "datasets"
    / "tarumt_dataset.csv"
)

MODEL_DIR = BASE_DIR / "model"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


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


# 4. Split dataset for internal evaluation
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


# 5. Perform stratified cross-validation configuration
smallest_training_class = int(y_train.value_counts().min()) # finds the intent with the fewest examples.

if smallest_training_class < 2:
    raise ValueError(
        "At least one intent has fewer than two examples in the 80% training split. Add more examples before using cross-validation."
    )

cv_folds = min(5, smallest_training_class) # place all the intents into 5 folds for cross-validation

cv = StratifiedKFold(
    n_splits=cv_folds,
    shuffle=True,
    random_state=42,
)

print(f"Grid search will use {cv_folds}-fold stratified cross-validation.")
print()


# 6. Convert training text into TF-IDF features
tfidf_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2),], 
    # (1, 1) = unigrams only, Ex: 'Where is my exam timetable?' -> features: timetable
    # (1, 2) = unigrams + bigrams, -> features: timetable + exam timetable

    "tfidf__min_df": [1, 2,],
    # min_df = minimum document frequency
    # a term must appear in at least two different questions before becoming a TF-IDF feature.

    "tfidf__sublinear_tf": [False, True],
    # Ex: Can i see the exam exam exam exam timetable? 
    # If without sublinear TF, exam would brings stronger influence due to repeatedly seeing it.
    # If == True, repeated occurrences have diminishing influence.

} # test sevaral conbinations and determine which one is the best

c_values = [0.1, 1.0, 10.0] 
# Test different regularization strengths


# 7. Pipeline
def run_grid_search(model_name, estimator):
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("model", estimator),
    ])

    param_grid = {
        **tfidf_grid,
        "model__C": c_values,
    }

    print("=" * 60)
    print(f"{model_name.upper()} GRID SEARCH")
    print("=" * 60)

    search = GridSearchCV(
        estimator = pipeline,     # TF-IDF -> model
        param_grid=param_grid,    # tries all the combinations ([2 ngram options] * [2 min_df options] * [2 sublinear_tf options] * [3 C options]) = 24 confogurations for each model
        scoring="f1_macro",       # f1_macro calculates the F1 for each intent and gives them equal importance
        cv=cv,                    
        n_jobs=-1,
        refit=True,
        verbose=1,
        return_train_score=True,
    )

    search.fit(X_train, y_train)

    print("\nBest parameters:")
    print(search.best_params_)
    print(f"Best mean CV Macro F1: {search.best_score_:.4f}")
    print()

    return search


# 8. Tune Logistic Regression
logistic_search = run_grid_search("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42))


# 9. Tune LinearSVC
linearsvc_search = run_grid_search("LinearSVC", LinearSVC(random_state=42))


# 10. Evaluate best tuned configurations on the untouched 20%
def evaluate_internal(model_name, search):
    model = search.best_estimator_
    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test, 
        predictions,
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    print("=" * 60)
    print(f"{model_name.upper()} - Results")
    print("=" * 60)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print()

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    print("=" * 60)
    print(f"{model_name.upper()} DETAILED PREDICTIONS")
    print("=" * 60)

    results = pd.DataFrame({
        "question": data.loc[X_test.index, "question"],
        "actual_intent": y_test,
        "predicted_intent": predictions,
    })

    results["correct"] = (
        results["actual_intent"]
        == results["predicted_intent"]
    )

    print(results.to_string(index=False))
    print()

    return accuracy, macro_f1, results

logistic_accuracy, logistic_macro_f1, logistic_results = evaluate_internal("Logistic Regression", logistic_search)

linearsvc_accuracy, linearsvc_macro_f1, linearsvc_results = evaluate_internal("LinearSVC", linearsvc_search)


# 11. Compare tuned models
comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "LinearSVC",
    ],
    "CV Macro F1": [
        logistic_search.best_score_,
        linearsvc_search.best_score_,
    ],
    "Internal Test Accuracy": [
        logistic_accuracy,
        linearsvc_accuracy,
    ],
    "Internal Test Macro F1": [
        logistic_macro_f1,
        linearsvc_macro_f1,
    ],
})

print("=" * 60)
print("TUNED MODEL COMPARISON")
print("=" * 60)
print(comparison.to_string(index=False))
print()

# Select student-facing model using untouched internal Macro F1. (Macro F1 -> balanced performance across all intents)
# Accuracy is used only as a tie-breaker. (Accuracy -> overall percentage correct)
if linearsvc_macro_f1 > logistic_macro_f1:
    selected_model = "LinearSVC"

elif logistic_macro_f1 > linearsvc_macro_f1:
    selected_model = "Logistic Regression"

elif linearsvc_accuracy > logistic_accuracy:
    selected_model = "LinearSVC"

elif logistic_accuracy > linearsvc_accuracy:
    selected_model = "Logistic Regression"

else:
    # Prefer LinearSVC on a complete tie because it is a strong baseline
    # for sparse high-dimensional text classification.
    selected_model = "LinearSVC"


print("Selected model:", selected_model)
print()


# 12. Save cross validation results for reporting
def save_cv_results(search, filename):
    cv_results = pd.DataFrame(search.cv_results_)

    useful_columns = [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
        "param_tfidf__ngram_range",
        "param_tfidf__min_df",
        "param_tfidf__sublinear_tf",
        "param_model__C",
    ]

    cv_results = cv_results[useful_columns].sort_values(
        ["rank_test_score", "mean_test_score"],
        ascending=[True, False],
    )

    cv_results.to_csv(
        MODEL_DIR / filename,
        index=False,
    )


save_cv_results(
    logistic_search,
    "logistic_cv_results.csv",
)

save_cv_results(
    linearsvc_search,
    "linearsvc_cv_results.csv",
)


# 13. Retrain final models using complete dataset
final_logistic_pipeline = clone(
    logistic_search.best_estimator_
)

final_linearsvc_pipeline = clone(
    linearsvc_search.best_estimator_
)

final_logistic_pipeline.fit(X, y)
final_linearsvc_pipeline.fit(X, y)


# 14. Save complete pipelines
joblib.dump(
    final_logistic_pipeline,
    MODEL_DIR / "logistic_pipeline.joblib",
)

joblib.dump(
    final_linearsvc_pipeline,
    MODEL_DIR / "linearsvc_pipeline.joblib",
)

(MODEL_DIR / "selected_model.txt").write_text(
    selected_model,
    encoding="utf-8",
)

print("Final models retrained using the complete dataset.")
print("Models saved successfully in the model folder.")


# 15. Save brief tuning summary
summary = pd.DataFrame([
    {
        "Model": "Logistic Regression",
        "CV Folds": cv_folds,
        "Best CV Macro F1": logistic_search.best_score_,
        "Internal Accuracy": logistic_accuracy,
        "Internal Macro F1": logistic_macro_f1,
        "Best ngram_range": str(
            logistic_search.best_params_["tfidf__ngram_range"]
        ),
        "Best min_df": logistic_search.best_params_["tfidf__min_df"],
        "Best sublinear_tf": logistic_search.best_params_[
            "tfidf__sublinear_tf"
        ],
        "Best C": logistic_search.best_params_["model__C"],
        "Selected": selected_model == "Logistic Regression",
    },
    {
        "Model": "LinearSVC",
        "CV Folds": cv_folds,
        "Best CV Macro F1": linearsvc_search.best_score_,
        "Internal Accuracy": linearsvc_accuracy,
        "Internal Macro F1": linearsvc_macro_f1,
        "Best ngram_range": str(
            linearsvc_search.best_params_["tfidf__ngram_range"]
        ),
        "Best min_df": linearsvc_search.best_params_["tfidf__min_df"],
        "Best sublinear_tf": linearsvc_search.best_params_[
            "tfidf__sublinear_tf"
        ],
        "Best C": linearsvc_search.best_params_["model__C"],
        "Selected": selected_model == "LinearSVC",
    },
])

summary.to_csv(
    MODEL_DIR / "tuning_summary.csv",
    index=False,
)

logistic_results.to_csv(
    MODEL_DIR / "logistic_internal_predictions.csv",
    index=False,
)

linearsvc_results.to_csv(
    MODEL_DIR / "linearsvc_internal_predictions.csv",
    index=False,
)
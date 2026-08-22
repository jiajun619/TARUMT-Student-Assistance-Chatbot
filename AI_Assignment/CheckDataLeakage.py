from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Project paths
BASE_DIR = Path(__file__).resolve().parent

TRAIN_PATH = BASE_DIR / "datasets" / "tarumt_dataset.csv"
UNSEEN_PATH = BASE_DIR / "datasets" / "unseen_test.csv"


# Load datasets
train_data = pd.read_csv(TRAIN_PATH)
unseen_data = pd.read_csv(UNSEEN_PATH)


# Normalize questions for exact duplicate checking
train_questions_normalized = (
    train_data["question"]
    .astype(str)
    .str.lower()
    .str.strip()
)

unseen_questions_normalized = (
    unseen_data["question"]
    .astype(str)
    .str.lower()
    .str.strip()
)


# Exact duplicate check
duplicates = unseen_data[
    unseen_questions_normalized.isin(
        set(train_questions_normalized)
    )
]


print("=" * 70)
print("DATA LEAKAGE CHECK")
print("=" * 70)

print("Training questions:", len(train_data))
print("Unseen questions:", len(unseen_data))
print("Exact duplicate questions:", len(duplicates))
print()


if duplicates.empty:
    print("No exact duplicate questions were found.")
else:
    print("WARNING: Exact duplicate questions were found:")
    print()

    print(
        duplicates[
            [
                "question",
                "expected_intent",
            ]
        ].to_string(index=False)
    )

print()


# TF-IDF similarity check
all_questions = pd.concat(
    [
        train_data["question"],
        unseen_data["question"],
    ],
    ignore_index=True,
)

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
)

all_vectors = vectorizer.fit_transform(all_questions)

train_vectors = all_vectors[:len(train_data)]
unseen_vectors = all_vectors[len(train_data):]


# Compare every unseen question with all training questions
similarity_matrix = cosine_similarity(
    unseen_vectors,
    train_vectors,
)


results = []

for unseen_index, similarities in enumerate(similarity_matrix):

    best_train_index = similarities.argmax()
    highest_similarity = similarities[best_train_index]

    results.append({
        "unseen_question":
            unseen_data.iloc[unseen_index]["question"],

        "expected_intent":
            unseen_data.iloc[unseen_index]["expected_intent"],

        "most_similar_training_question":
            train_data.iloc[best_train_index]["question"],

        "training_intent":
            train_data.iloc[best_train_index]["intent"],

        "similarity":
            highest_similarity,
    })


similarity_results = pd.DataFrame(results)

similarity_results = similarity_results.sort_values(
    by="similarity",
    ascending=False,
)


print("=" * 70)
print("TF-IDF SIMILARITY CHECK")
print("=" * 70)

print(
    similarity_results.to_string(
        index=False,
        formatters={
            "similarity": lambda value: f"{value:.4f}"
        },
    )
)

print()


# Flag highly similar questions
HIGH_SIMILARITY_THRESHOLD = 0.90

high_similarity_cases = similarity_results[
    similarity_results["similarity"]
    >= HIGH_SIMILARITY_THRESHOLD
]


print("=" * 70)
print(
    f"HIGH SIMILARITY CASES "
    f"(Similarity >= {HIGH_SIMILARITY_THRESHOLD})"
)
print("=" * 70)

print(
    "Number of highly similar unseen questions:",
    len(high_similarity_cases),
)
print()


if high_similarity_cases.empty:
    print(
        "No unseen questions have extremely high "
        "similarity with the training dataset."
    )
else:
    print(
        high_similarity_cases.to_string(
            index=False,
            formatters={
                "similarity": lambda value: f"{value:.4f}"
            },
        )
    )
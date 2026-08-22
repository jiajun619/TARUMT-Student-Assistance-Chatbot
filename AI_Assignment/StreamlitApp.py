from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from Chatbot import chatbot
from NLP import rejoin


# Page configuration
st.set_page_config(
    page_title="TARUMT Student Assistance Chatbot",
    page_icon="🎓",
    layout="wide",
)


# Project paths
BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "model"
UNSEEN_DATASET_PATH = BASE_DIR / "datasets" / "unseen_test.csv"


# Load models
@st.cache_resource
def load_models():
    vectorizer = joblib.load(
        MODEL_DIR / "vectorizer.joblib"
    )

    logistic_model = joblib.load(
        MODEL_DIR / "logistic_model.joblib"
    )

    linearsvc_model = joblib.load(
        MODEL_DIR / "linearsvc_model.joblib"
    )

    return vectorizer, logistic_model, linearsvc_model


# Evaluate models using unseen dataset
@st.cache_data
def evaluate_models():
    test_data = pd.read_csv(UNSEEN_DATASET_PATH)

    test_data["processed_question"] = (
        test_data["question"].apply(rejoin)
    )

    X_test = test_data["processed_question"]
    y_test = test_data["expected_intent"]

    vectorizer, logistic_model, linearsvc_model = load_models()

    X_test_vectorized = vectorizer.transform(X_test)

    # Logistic Regression predictions
    logistic_predictions = logistic_model.predict(
        X_test_vectorized
    )

    # LinearSVC predictions
    linearsvc_predictions = linearsvc_model.predict(
        X_test_vectorized
    )

    # Logistic Regression metrics
    logistic_report = classification_report(
        y_test,
        logistic_predictions,
        output_dict=True,
        zero_division=0,
    )

    logistic_accuracy = accuracy_score(
        y_test,
        logistic_predictions,
    )

    # LinearSVC metrics
    linearsvc_report = classification_report(
        y_test,
        linearsvc_predictions,
        output_dict=True,
        zero_division=0,
    )

    linearsvc_accuracy = accuracy_score(
        y_test,
        linearsvc_predictions,
    )

    # Comparison table
    comparison = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score",
        ],
        "Logistic Regression": [
            logistic_accuracy,
            logistic_report["macro avg"]["precision"],
            logistic_report["macro avg"]["recall"],
            logistic_report["macro avg"]["f1-score"],
        ],
        "LinearSVC": [
            linearsvc_accuracy,
            linearsvc_report["macro avg"]["precision"],
            linearsvc_report["macro avg"]["recall"],
            linearsvc_report["macro avg"]["f1-score"],
        ],
    })

    # Confusion matrices
    labels = sorted(y_test.unique())

    logistic_cm = confusion_matrix(
        y_test,
        logistic_predictions,
        labels=labels,
    )

    linearsvc_cm = confusion_matrix(
        y_test,
        linearsvc_predictions,
        labels=labels,
    )

    logistic_cm_df = pd.DataFrame(
        logistic_cm,
        index=labels,
        columns=labels,
    )

    linearsvc_cm_df = pd.DataFrame(
        linearsvc_cm,
        index=labels,
        columns=labels,
    )

    # Detailed prediction results
    logistic_results = pd.DataFrame({
        "Question": test_data["question"],
        "Expected Intent": y_test,
        "Predicted Intent": logistic_predictions,
    })

    logistic_results["Correct"] = (
        logistic_results["Expected Intent"]
        == logistic_results["Predicted Intent"]
    )

    linearsvc_results = pd.DataFrame({
        "Question": test_data["question"],
        "Expected Intent": y_test,
        "Predicted Intent": linearsvc_predictions,
    })

    linearsvc_results["Correct"] = (
        linearsvc_results["Expected Intent"]
        == linearsvc_results["Predicted Intent"]
    )

    return (
        comparison,
        logistic_cm_df,
        linearsvc_cm_df,
        logistic_results,
        linearsvc_results,
    )


# Title
st.title("🎓 TARUMT Student Assistance Chatbot")

st.caption(
    "Natural Language Processing chatbot using "
    "Logistic Regression and LinearSVC."
)


# Tabs
chat_tab, comparison_tab = st.tabs(
    [
        "💬 Chatbot",
        "📊 Model Comparison",
    ]
)


# Chatbot tab
with chat_tab:

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        model_name = st.radio(
            "Select Model",
            [
                "Logistic Regression",
                "LinearSVC",
            ],
            index=1,
        )

        st.divider()

        st.write("Current Model:")
        st.success(model_name)

        if st.button("Clear Chat"):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": (
                        "Hello! I am the TARUMT Student Assistance Chatbot. "
                        "How can I help you today?"
                    ),
                }
            ]

            st.rerun()

    # Reset chat when model changes
    if "previous_model" not in st.session_state:
        st.session_state.previous_model = model_name

    if st.session_state.previous_model != model_name:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! I am the TARUMT Student Assistance Chatbot. "
                    "How can I help you today?"
                ),
            }
        ]

        st.session_state.previous_model = model_name
        st.rerun()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! I am the TARUMT Student Assistance Chatbot. "
                    "How can I help you today?"
                ),
            }
        ]

    st.write(
        "Ask me about admissions, timetables, examinations, "
        "fees, scholarships, programmes, and campus facilities."
    )

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User input
    user_message = st.chat_input(
        "Type your question here..."
    )

    if user_message:
        st.session_state.messages.append({
            "role": "user",
            "content": user_message,
        })

        with st.chat_message("user"):
            st.write(user_message)

        try:
            response, intent, score = chatbot(
                user_message,
                model_name,
            )

            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
            })

            with st.chat_message("assistant"):
                st.write(response)

                with st.expander("Prediction Details"):
                    st.write(
                        f"**Model:** {model_name}"
                    )

                    st.write(
                        f"**Predicted Intent:** {intent}"
                    )

                    if model_name == "Logistic Regression":
                        st.write(
                            f"**Probability:** {score:.4f}"
                        )
                    else:
                        st.write(
                            f"**Decision Score:** {score:.4f}"
                        )

        except Exception as error:
            st.error(
                "Sorry, something went wrong."
            )

            st.write(error)


# Model comparison tab
with comparison_tab:

    st.header("📊 Model Performance Comparison")

    st.write(
        "The two trained models are evaluated using "
        "the separate unseen test dataset."
    )

    (
        comparison,
        logistic_cm,
        linearsvc_cm,
        logistic_results,
        linearsvc_results,
    ) = evaluate_models()

    # Evaluation metrics
    st.subheader("Evaluation Metrics")

    display_comparison = comparison.copy()

    display_comparison[
        [
            "Logistic Regression",
            "LinearSVC",
        ]
    ] = (
        display_comparison[
            [
                "Logistic Regression",
                "LinearSVC",
            ]
        ] * 100
    ).round(2)

    display_comparison = display_comparison.rename(
        columns={
            "Logistic Regression":
                "Logistic Regression (%)",
            "LinearSVC":
                "LinearSVC (%)",
        }
    )

    st.dataframe(
        display_comparison,
        use_container_width=True,
        hide_index=True,
    )

    # Accuracy summary
    logistic_accuracy = comparison.loc[
        comparison["Metric"] == "Accuracy",
        "Logistic Regression",
    ].iloc[0]

    linearsvc_accuracy = comparison.loc[
        comparison["Metric"] == "Accuracy",
        "LinearSVC",
    ].iloc[0]

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Logistic Regression Accuracy",
            f"{logistic_accuracy * 100:.2f}%",
        )

    with col2:
        st.metric(
            "LinearSVC Accuracy",
            f"{linearsvc_accuracy * 100:.2f}%",
        )

    # Best model
    if logistic_accuracy > linearsvc_accuracy:
        st.success(
            "Logistic Regression achieved the higher "
            "accuracy on the unseen test dataset."
        )

    elif linearsvc_accuracy > logistic_accuracy:
        st.success(
            "LinearSVC achieved the higher accuracy "
            "on the unseen test dataset."
        )

    else:
        st.info(
            "Both models achieved the same accuracy "
            "on the unseen test dataset."
        )

    # Confusion matrices
    st.subheader("Confusion Matrices")

    st.write("**Logistic Regression**")

    st.dataframe(
        logistic_cm,
        use_container_width=True,
    )

    st.write("**LinearSVC**")

    st.dataframe(
        linearsvc_cm,
        use_container_width=True,
    )

    # Detailed predictions
    st.subheader("Detailed Predictions")

    model_result = st.selectbox(
        "Select prediction results to view",
        [
            "Logistic Regression",
            "LinearSVC",
        ],
    )

    if model_result == "Logistic Regression":
        st.dataframe(
            logistic_results,
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.dataframe(
            linearsvc_results,
            use_container_width=True,
            hide_index=True,
        )
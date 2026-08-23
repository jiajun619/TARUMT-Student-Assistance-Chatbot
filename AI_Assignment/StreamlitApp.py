from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

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


# Greeting message
GREETING_MESSAGE = (
    "Hello! I am the TARUMT Student Assistance Chatbot. "
    "How can I help you today?"
)


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


# Student-facing settings
STUDENT_MODEL = "LinearSVC"

NEXT_ACTIONS = {
    "admission": (
        "Prepare the required documents and use the TARUMT online application "
        "system. Check your application status and email for updates."
    ),
    "timetable": "Open TARC App → Class Timetable.",
    "examination": (
        "Open TARC App and check the Examination section for your exam "
        "timetable or results."
    ),
    "fees": "Open TARC App → Billing → Current Billing.",
    "scholarship": (
        "Check TARUMT's official scholarship information and review the "
        "eligibility and application requirements."
    ),
    "programme": "Open Student Intranet → Programme → Programme Structure.",
    "campus_facility": (
        "Open TARC App → Campus Map. For the latest facility operating hours, "
        "check the official campus information or contact your campus."
    ),
}

FEEDBACK_INTENTS = {
    "admission",
    "timetable",
    "examination",
    "fees",
    "scholarship",
    "programme",
    "campus_facility",
    "unknown",
}

QUICK_HELP_ITEMS = [
    {
        "title": "📅 Timetable",
        "description": "Check your class schedule and lesson times.",
        "question": "How do I check my class timetable?",
        "key": "quick_timetable",
    },
    {
        "title": "💳 Fees",
        "description": "Find billing and outstanding fee information.",
        "question": "Where can I check my current fees?",
        "key": "quick_fees",
    },
    {
        "title": "📝 Examination",
        "description": "Check exam timetable, venue, and results.",
        "question": "Where can I find my exam timetable?",
        "key": "quick_examination",
    },
    {
        "title": "🎓 Admission",
        "description": "Get guidance on TARUMT applications.",
        "question": "How do I apply to TARUMT?",
        "key": "quick_admission",
    },
]

FALLBACK_QUESTIONS = [
    (
        "📅 Timetable",
        "Where can I check my class timetable?",
        "timetable",
    ),
    (
        "💳 Fees",
        "Where can I check my current fees?",
        "fees",
    ),
    (
        "🎓 Admission",
        "How do I apply to TARUMT?",
        "admission",
    ),
]

MODEL_NAMES = [
    "Logistic Regression",
    "LinearSVC",
]


def reset_chat():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": GREETING_MESSAGE,
        }
    ]

def reset_feedback():
    st.session_state.feedback_records = []

    for message in st.session_state.get("messages", []):
        if "feedback" in message:
            message["feedback"] = None

def process_user_message(message):
    if not message or not message.strip():
        return

    st.session_state.messages.append({
        "role": "user",
        "content": message,
    })

    try:
        response, intent, _ = chatbot(
            message,
            STUDENT_MODEL,
        )

        assistant_message = {
            "role": "assistant",
            "content": response,
            "intent": intent,
            "feedback": None,
        }

        next_action = NEXT_ACTIONS.get(intent)

        if next_action:
            assistant_message["next_action"] = next_action

        st.session_state.messages.append(assistant_message)

    except Exception:
        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                "Sorry, something went wrong while processing your question. "
                "Please try again."
            ),
        })


def submit_question(message):
    process_user_message(message)

    st.session_state.scroll_trigger = (
        st.session_state.get("scroll_trigger", 0) + 1
    )
    st.session_state.scroll_to_latest = True
    st.rerun()


def record_feedback(message_index, intent, feedback):
    st.session_state.messages[message_index]["feedback"] = feedback
    st.session_state.feedback_records.append({
        "intent": intent,
        "feedback": feedback,
    })
    st.rerun()


def scroll_chat_to_bottom(trigger_id):
    components.html(
        f"""
        <script>
            const triggerId = {trigger_id};
            const doc = window.parent.document;
            const win = window.parent;

            function findScrollableParent(element) {{
                let current = element.parentElement;

                while (current) {{
                    const style = win.getComputedStyle(current);

                    const canScroll =
                        current.scrollHeight > current.clientHeight;

                    const overflowAllowsScroll =
                        style.overflowY === "auto" ||
                        style.overflowY === "scroll" ||
                        style.overflowY === "overlay";

                    if (canScroll && overflowAllowsScroll) {{
                        return current;
                    }}

                    current = current.parentElement;
                }}

                return null;
            }}

            function scrollEverything() {{
                const chatAnchor =
                    doc.getElementById("chat-bottom-anchor");

                const pageAnchor =
                    doc.getElementById("chat-page-anchor");

                if (!chatAnchor || !pageAnchor) {{
                    return false;
                }}

                // Scroll inside chat box
                const chatScroller =
                    findScrollableParent(chatAnchor);

                if (chatScroller) {{
                    chatScroller.scrollTop =
                        chatScroller.scrollHeight;
                }}

                // Scroll Streamlit outer page
                const pageScroller =
                    findScrollableParent(pageAnchor);

                if (pageScroller) {{
                    pageScroller.scrollTo({{
                        top: pageScroller.scrollHeight,
                        behavior: "smooth"
                    }});
                }} else {{
                    pageAnchor.scrollIntoView({{
                        behavior: "smooth",
                        block: "end"
                    }});
                }}

                return true;
            }}

            let attempts = 0;

            const timer = setInterval(() => {{
                attempts++;

                if (scrollEverything() || attempts >= 20) {{
                    clearInterval(timer);
                }}
            }}, 100);
        </script>
        <!-- scroll-trigger-{trigger_id} -->
        """,
        height=0,
    )

# Initialize feedback records
if "feedback_records" not in st.session_state:
    st.session_state.feedback_records = []

# Sidebar navigation
with st.sidebar:
    st.header("🎓 Navigation")

    page = st.radio(
        "Select Page",
        [
            "🎓 Student Chatbot",
            "📊 Technical Evaluation",
        ],
    )

    st.divider()

    if page == "🎓 Student Chatbot":
        st.subheader("About")
        st.write(
            "This chatbot helps students find common TARUMT information "
            "and the next step to take."
        )

        if st.button("Clear Chat", use_container_width=True):
            reset_chat()
            st.rerun()


# Student chatbot page
if page == "🎓 Student Chatbot":
    st.title("🎓 TARUMT Student Assistance Chatbot")

    st.caption(
        "Ask a TARUMT-related question and get guidance on what to do next."
    )

    if "messages" not in st.session_state:
        reset_chat()

    st.subheader("Quick Help")

    quick_cols = st.columns(len(QUICK_HELP_ITEMS))

    for col, item in zip(quick_cols, QUICK_HELP_ITEMS):
        with col:
            with st.container(border=True):
                st.markdown(f"**{item['title']}**")
                st.caption(item["description"])

                if st.button(
                    "Ask chatbot",
                    key=item["key"],
                    use_container_width=True,
                ):
                    submit_question(item["question"])

    st.caption(
        "You can also ask about scholarships, programmes, and campus facilities."
    )

    # Scrollable chat history
    chat_container = st.container(
        height=520,
        border=True,
    )

    with chat_container:
        for message_index, message in enumerate(
            st.session_state.messages
        ):
            with st.chat_message(message["role"]):
                st.write(message["content"])

                if "next_action" in message:
                    st.markdown("**👉 What you can do next**")
                    st.info(message["next_action"])

                if (
                    message["role"] == "assistant"
                    and message.get("intent") == "unknown"
                ):
                    st.caption("Try one of these common questions:")

                    fallback_cols = st.columns(len(FALLBACK_QUESTIONS))

                    for col, (label, question, key) in zip(
                        fallback_cols,
                        FALLBACK_QUESTIONS,
                    ):
                        with col:
                            if st.button(
                                label,
                                key=f"fallback_{key}_{message_index}",
                                use_container_width=True,
                            ):
                                submit_question(question)

                # User feedback for chatbot responses
                if (
                    message["role"] == "assistant"
                    and message.get("intent") in FEEDBACK_INTENTS
                ):
                    feedback = message.get("feedback")

                    if feedback is None:
                        st.caption("Was this response helpful?")

                        feedback_col1, feedback_col2 = st.columns(
                            [1, 1]
                        )

                        with feedback_col1:
                            if st.button(
                                "👍 Helpful",
                                key=f"helpful_{message_index}",
                                use_container_width=True,
                            ):
                                record_feedback(
                                    message_index,
                                    message.get("intent"),
                                    "helpful",
                                )

                        with feedback_col2:
                            if st.button(
                                "👎 Not Helpful",
                                key=f"not_helpful_{message_index}",
                                use_container_width=True,
                            ):
                                record_feedback(
                                    message_index,
                                    message.get("intent"),
                                    "not_helpful",
                                )

                    else:
                        st.caption(
                            "✅ Thank you for your feedback!"
                        )

        # Bottom of chat history
        st.markdown(
            '<div id="chat-bottom-anchor"></div>',
            unsafe_allow_html=True,
        )

    # Marker for automatic scrolling
    st.markdown(
        '<div id="chat-page-anchor"></div>',
        unsafe_allow_html=True,
    )


    # Automatically scroll to latest response
    if st.session_state.pop("scroll_to_latest", False):
        scroll_chat_to_bottom(
            st.session_state.get("scroll_trigger", 0)
        )

    # Fixed chat input
    user_message = st.chat_input(
        "Ask a TARUMT question..."
    )

    if user_message:
        submit_question(user_message)


# Technical evaluation page
elif page == "📊 Technical Evaluation":
    st.title("📊 Technical Evaluation")

    st.caption(
        "Model evaluation for Logistic Regression and LinearSVC using the "
        "separate unseen test dataset."
    )

    st.info(
        "This page is for evaluating the AI models. The Student Chatbot page "
        "is the main user-facing prototype."
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
            "Logistic Regression": "Logistic Regression (%)",
            "LinearSVC": "LinearSVC (%)",
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
            "Logistic Regression achieved the higher accuracy on the unseen "
            "test dataset."
        )

    elif linearsvc_accuracy > logistic_accuracy:
        st.success(
            "LinearSVC achieved the higher accuracy on the unseen test dataset."
        )

    else:
        st.info(
            "Both models achieved the same accuracy on the unseen test dataset."
        )

    # User feedback summary
    st.subheader("User Feedback")

    if st.button(
        "Reset Feedback",
        use_container_width=False,
    ):
        reset_feedback()
        st.rerun()

    feedback_df = pd.DataFrame(
        st.session_state.get("feedback_records", [])
    )

    if not feedback_df.empty:
        total_feedback = len(feedback_df)
        helpful_feedback = (
            feedback_df["feedback"] == "helpful"
        ).sum()
        not_helpful_feedback = (
            feedback_df["feedback"] == "not_helpful"
        ).sum()
        satisfaction_rate = (
            helpful_feedback / total_feedback
        ) * 100

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Responses Rated", total_feedback)

        with col2:
            st.metric("Helpful", helpful_feedback)

        with col3:
            st.metric(
                "Satisfaction Rate",
                f"{satisfaction_rate:.2f}%",
            )

        st.write(f"Not Helpful: **{not_helpful_feedback}**")

        # Feedback breakdown by intent
        feedback_summary = (
            feedback_df
            .groupby("intent")
            .agg(
                Rated=("feedback", "count"),
                Helpful=(
                    "feedback",
                    lambda x: (x == "helpful").sum()
                ),
                Not_Helpful=(
                    "feedback",
                    lambda x: (x == "not_helpful").sum()
                ),
            )
            .reset_index()
        )

        feedback_summary["Satisfaction (%)"] = (
            feedback_summary["Helpful"]
            / feedback_summary["Rated"]
            * 100
        ).round(2)

        feedback_summary = feedback_summary.rename(
            columns={
                "intent": "Intent",
                "Not_Helpful": "Not Helpful",
            }
        )

        st.write("**Feedback by Intent**")

        st.dataframe(
            feedback_summary,
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info(
            "No chatbot responses have been rated "
            "in the current session yet."
        )

    # Error analysis
    st.subheader("Error Analysis")

    logistic_errors = logistic_results[
        logistic_results["Correct"] == False
    ]

    linearsvc_errors = linearsvc_results[
        linearsvc_results["Correct"] == False
    ]

    error_col1, error_col2 = st.columns(2)

    with error_col1:
        st.metric(
            "Logistic Regression Errors",
            len(logistic_errors),
        )

    with error_col2:
        st.metric(
            "LinearSVC Errors",
            len(linearsvc_errors),
        )

    with st.expander("View Misclassified Cases"):
        error_model = st.selectbox(
            "Select model",
            MODEL_NAMES,
            key="error_analysis_model",
        )

        if error_model == "Logistic Regression":
            selected_errors = logistic_errors
        else:
            selected_errors = linearsvc_errors

        if selected_errors.empty:
            st.success(
                "No misclassified cases were found for this model."
            )
        else:
            st.dataframe(
                selected_errors[
                    [
                        "Question",
                        "Expected Intent",
                        "Predicted Intent",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

    # Confusion matrices
    with st.expander("View Confusion Matrices"):
        st.caption(
            "Rows represent the actual intent and columns represent the predicted intent. "
            "Higher values on the diagonal indicate correct predictions."
        )

        cm_model = st.selectbox(
            "Select model",
            MODEL_NAMES,
            key="confusion_matrix_model",
        )

        if cm_model == "Logistic Regression":
            selected_cm = logistic_cm.copy()
        else:
            selected_cm = linearsvc_cm.copy()

        selected_cm.index.name = "Actual"
        selected_cm.columns.name = "Predicted"

        styled_cm = (
            selected_cm.style
            .background_gradient(
                cmap="Blues",
                axis=None,
            )
            .format("{:.0f}")
        )

        st.dataframe(
            styled_cm,
            use_container_width=True,
        )

    # Detailed predictions
    with st.expander("View Detailed Predictions"):
        model_result = st.selectbox(
            "Select prediction results to view",
            MODEL_NAMES,
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
import streamlit as st
from Chatbot import chatbot

st.set_page_config(
    page_title="TARUMT Student Assistance Chatbot",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 TARUMT Student Assistance Chatbot")

st.caption(
    "Ask me about admissions, timetables, examinations, fees, "
    "scholarships, programmes, and campus facilities."
)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    model_name = st.radio(
        "Select Model",
        ["Logistic Regression", "LinearSVC"],
        index=1
    )

    st.divider()

    st.write("Current Model:")
    st.success(model_name)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! I am the TARUMT Student Assistance Chatbot. "
                "How can I help you today?"
            )
        }
    ]

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
user_message = st.chat_input("Type your question here...")

if user_message:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_message
        }
    )

    with st.chat_message("user"):
        st.write(user_message)

    try:
        response, intent, score = chatbot(
            user_message,
            model_name
        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )

        with st.chat_message("assistant"):
            st.write(response)

            with st.expander("Prediction Details"):
                st.write(f"**Model:** {model_name}")
                st.write(f"**Predicted Intent:** {intent}")

                if model_name == "Logistic Regression":
                    st.write(f"**Probability:** {score:.4f}")
                else:
                    st.write(f"**Decision Score:** {score:.4f}")

    except Exception as error:
        st.error("Sorry, something went wrong.")
        st.write(error)
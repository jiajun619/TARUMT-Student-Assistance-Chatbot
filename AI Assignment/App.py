from flask import Flask, render_template, request, jsonify
from Chatbot import chatbot

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({
            "response": "Please enter a question."
        }), 400

    try:
        response = chatbot(message)

        return jsonify({
            "response": response
        })

    except Exception as error:
        print("Chatbot error:", error)

        return jsonify({
            "response": "Sorry, something went wrong. Please try again."
        }), 500


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
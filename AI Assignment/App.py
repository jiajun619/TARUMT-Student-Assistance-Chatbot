from flask import Flask,render_template,request,jsonify

from Chatbot import chatbot

app=Flask(__name__)

@app.route("/")

def home():

    return render_template("index.html")

@app.route("/chat",methods=["POST"])

def chat():

    data=request.get_json()

    message=data["message"]

    response=chatbot(message)

    return jsonify({

        "response":response

    })

if __name__=="__main__":

    app.run(debug=True)
from flask import Flask,request,jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__) # creating a instance of flask

@app.route("/",methods=["GET"]) # descorator: function to be executed when someone visits
def home():
    return "CodeVeDX chatbot API is running"
@app.route("/echo",methods=["POST"]) # receiving or sending a data
def echo(): # function runs when someone posts to echo
    data = request.get_json() # parse the incoming json body
    if not data or "message" not in data: # if the data is absent or message field not present , then
        # return a error with http status 400 [ bad request ] 
        return jsonify({"error":"Please send a messagge field in json"}),400 
    return jsonify({"you:>":data["message"]}) # else send back the message

intent_keywords = {
    "password_reset": ["password", "reset", "forgot", "pwd"],
    "wifi_access": ["wifi", "wi-fi", "network", "internet"],
    "leave_request": ["leave", "vacation", "time off", "holiday"],
    "ticket_status": ["ticket", "status", "issue", "complaint"],
}

def detect_intent(text):
    text = text.lower()
    for intent,keyword in intent_keywords.items():
        for key in keyword:
            if key in text:
                return intent
    return "Unknown"

faq_df = pd.read_csv("helpdesk_faq.csv")
tfidf = TfidfVectorizer(stop_words="english") # ignore the common filler words
faq_vectors = tfidf.fit_transform(faq_df["question"]) # convert all the faq questions into tfidf numeric vectors

def find_best_answer(query):
    query_vector = tfidf.transform([query])
    similarities = cosine_similarity(query_vector,faq_vectors)
    best_match_index = similarities.argmax() # return the index of the maximum element [most matched element]
    best_score = similarities.max()
    if best_score < 0.2:
        return "Sorry , I couldnt find a relevant answer.Next?Contact Support!"
    return faq_df.iloc[best_match_index]["answer"]

@app.route("/chat",methods=["POST"]) # main chatbot route
def chat():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error":"Please send a message field in json"}),400
    user_message = data["message"]
    intent = detect_intent(user_message)
    answer = find_best_answer(user_message)
    return jsonify({"user:":user_message,"detected_Intent":intent,"CodeVedX:":answer})

# Only run the server if the app is launched via this code
if __name__=="__main__":
    # start the flash development server on port 5000 with auto-reload
    app.run(debug=True,port=5000)
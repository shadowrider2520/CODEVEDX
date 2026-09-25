# import Flask core, request handling, and JSON response helper
from flask import Flask, request, jsonify
# import pandas to load and work with the FAQ dataset
import pandas as pd
# import TfidfVectorizer to convert text into numeric vectors
from sklearn.feature_extraction.text import TfidfVectorizer
# import cosine_similarity to measure how close two texts are
from sklearn.metrics.pairwise import cosine_similarity

# create the Flask application instance
app = Flask(__name__)
# a simple shared secret key used to authorize admin-only actions
ADMIN_KEY = "codevedx_admin_2026"

# dictionary mapping each intent to its trigger keywords
intent_keywords = {
    "password_reset": ["password", "reset", "forgot", "pwd"],
    "wifi_access": ["wifi", "wi-fi", "network", "internet"],
    "leave_request": ["leave", "vacation", "time off", "holiday"],
    "ticket_status": ["ticket", "status", "issue", "complaint"],
}

# function that detects the user's intent from their message
def detect_intent(text):
    # convert text to lowercase so matching isn't case-sensitive
    text = text.lower()
    # loop through each intent and its list of keywords
    for intent, keywords in intent_keywords.items():
        # check if any keyword from this intent appears in the text
        if any(k in text for k in keywords):
            # return this intent as soon as a match is found
            return intent
    # if no keywords matched anything, return "unknown"
    return "unknown"

# load the FAQ dataset from disk
faq_df = pd.read_csv("helpdesk_faq.csv")
# create a TF-IDF vectorizer, ignoring common filler words
tfidf = TfidfVectorizer(stop_words="english")
# convert all FAQ questions into TF-IDF numeric vectors
faq_vectors = tfidf.fit_transform(faq_df["question"])

# function that checks if the request has the correct admin key
def is_authorized(req):
    # compare the custom header value against our stored admin key
    return req.headers.get("X-Admin-Key") == ADMIN_KEY

# function that rebuilds the TF-IDF index after FAQ data changes
def rebuild_index():
    # tell Python we're modifying these global variables, not local ones
    global tfidf, faq_vectors
    # create a fresh vectorizer so old/new vocabulary stays in sync
    tfidf = TfidfVectorizer(stop_words="english")
    # refit it on the current (possibly updated) FAQ questions
    faq_vectors = tfidf.fit_transform(faq_df["question"])

# function that finds the best matching FAQ answer for a user query
def find_best_answer(query):
    # convert the user's query into a TF-IDF vector (wrapped in a list)
    vec = tfidf.transform([query])
    # compare the user's query vector against all FAQ question vectors
    sims = cosine_similarity(vec, faq_vectors)
    # if no FAQ is similar enough, admit we don't know
    if sims.max() < 0.2:
        # return a fallback message instead of a wrong answer
        return "Sorry, I couldn't find a relevant answer. Contact IT support."
    # otherwise return the answer for the closest matching question
    return faq_df.iloc[sims.argmax()]["answer"]

# define the main chatbot POST route
@app.route("/chat", methods=["POST"])
# function that runs when someone POSTs a message to "/chat"
def chat():
    # parse the incoming JSON body from the request
    data = request.get_json()
    # check if data exists and contains a "message" field
    if not data or "message" not in data:
        # return an error response with HTTP status 400 (bad request)
        return jsonify({"error": "message field required"}), 400
    # extract the actual user message from the request data
    msg = data["message"]
    # return both the detected intent and the best FAQ answer
    return jsonify({
        "intent": detect_intent(msg),
        "answer": find_best_answer(msg)
    })

# define a POST route for admins to add a new FAQ entry
@app.route("/admin/faq/add", methods=["POST"])
# function that runs when someone POSTs to "/admin/faq/add"
def add_faq():
    # check if the request is authorized before doing anything
    if not is_authorized(request):
        # return an error response with HTTP status 401 (unauthorized)
        return jsonify({"error": "Unauthorized"}), 401
    # parse the incoming JSON body from the request
    data = request.get_json()
    # check that both required fields are present
    if not data or "question" not in data or "answer" not in data:
        # return an error response with HTTP status 400 (bad request)
        return jsonify({"error": "question and answer required"}), 400

    # tell Python we're modifying this global variable, not a local one
    global faq_df
    # append the new question/answer row to the existing FAQ dataset
    faq_df = pd.concat([faq_df, pd.DataFrame([data])], ignore_index=True)
    # save the updated dataset back to the CSV file
    faq_df.to_csv("helpdesk_faq.csv", index=False)
    # rebuild the TF-IDF index so the new question becomes searchable
    rebuild_index()
    # confirm the FAQ was added and show the new total count
    return jsonify({"message": "FAQ added", "total": len(faq_df)})

# define a GET route for admins to view all current FAQs
@app.route("/admin/faq/list", methods=["GET"])
# function that runs when someone GETs "/admin/faq/list"
def list_faqs():
    # check if the request is authorized before showing data
    if not is_authorized(request):
        # return an error response with HTTP status 401 (unauthorized)
        return jsonify({"error": "Unauthorized"}), 401
    # convert the FAQ DataFrame into a list of dictionaries and return as JSON
    return jsonify(faq_df.to_dict(orient="records"))

# define a DELETE route for admins to remove an FAQ by its row index
@app.route("/admin/faq/delete/<int:index>", methods=["DELETE"])
# function that runs when someone sends DELETE to "/admin/faq/delete/<index>"
def delete_faq(index):
    # check if the request is authorized before deleting anything
    if not is_authorized(request):
        # return an error response with HTTP status 401 (unauthorized)
        return jsonify({"error": "Unauthorized"}), 401
    # tell Python we're modifying this global variable, not a local one
    global faq_df
    # check that the given index actually exists in the dataset
    if index >= len(faq_df):
        # return an error response with HTTP status 404 (not found)
        return jsonify({"error": "Invalid index"}), 404
    # remove the row at the given index and re-number the remaining rows
    faq_df = faq_df.drop(index).reset_index(drop=True)
    # save the updated dataset back to the CSV file
    faq_df.to_csv("helpdesk_faq.csv", index=False)
    # rebuild the TF-IDF index since the FAQ data has changed
    rebuild_index()
    # confirm the FAQ was deleted and show the new total count
    return jsonify({"message": "FAQ deleted", "total": len(faq_df)})

# only run the server if this file is executed directly
if __name__ == "__main__":
    # start the Flask development server on port 5000 with auto-reload
    app.run(debug=True, port=5000)
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

intent_keywords = {
    "password_reset": ["password", "reset", "forgot", "pwd"],
    "wifi_access": ["wifi", "wi-fi", "network", "internet"],
    "leave_request": ["leave", "vacation", "time off", "holiday"],
    "ticket_status": ["ticket", "status", "issue", "complaint"],
}
def detect_intent(text):
    text = text.lower()                                                        # normalize case first
    for intent, keywords in intent_keywords.items():                            # check each intent's keyword list
        if any(keyword in text for keyword in keywords):                          # if any keyword appears in the text
            return intent                                                          # return that intent
    return "unknown"

def extract_entities(text):
    entities = {}
    date_match = re.search(r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b', text)
    if date_match:
        entities["date"] = date_match.group()
    ticket_match = re.search(r'\b(ticket\s*#?\d+)\b', text,re.IGNORECASE)
    if ticket_match:
        entities["ticket_id"] = ticket_match.group()
    return entities
# ---- Test intent + entity detection together ----
test_queries = [
    "I forgot my password, can you reset it?",
    "Wifi is not working in the office today",
    "I need leave on 12/05/2026",
    "Please check ticket #4521 status",
    "what is my name ? "
]

print("--- Intent & Entity Detection ---\n")
for query in test_queries:
    intent = detect_intent(query)                                                     # figure out what they want
    entities = extract_entities(query)                                                  # pull out extra details
    print(f"Query: {query}\nIntent: {intent} | Entities: {entities}\n")

faq_data = {
    "question": [
        "How do I reset my password?",
        "How can I connect to the office wifi?",
        "How do I request leave?",
        "How do I check my IT ticket status?",
        "Who do I contact for laptop issues?",
        "What is my name?"
    ],
    "answer": [
        "Go to the IT portal, click 'Forgot Password', and follow the reset link sent to your email.",
        "Connect to 'CodeVedX_Office' network and use the password posted on the office noticeboard.",
        "Submit a leave request through the HR portal under the 'Apply Leave' section.",
        "Log into the helpdesk portal and search your ticket number under 'My Tickets'.",
        "Contact the IT support desk at extension 204 or raise a ticket on the helpdesk portal.",
        "Im mithun prasath"
    ]
}
faq_df = pd.DataFrame(faq_data)
faq_df.to_csv("helpdesk_FAQ.csv",index=False)
print("Dataset saved for helpdesk frequently asked qquestions\n")
tfidf = TfidfVectorizer(stop_words="english")
faq_vectors = tfidf.fit_transform(faq_df["question"])

def find_best_answer(query):
    query_vector = tfidf.transform([query])
    similarities = cosine_similarity(query_vector,faq_vectors)
    best_match_index = similarities.argmax()
    best_score = similarities.max()

    if best_score<0.3:
        return "Sorry i couldnt find the relavant answer,please contact support"
    return faq_df.iloc[best_match_index]["answer"]

print("------FAQ matching Demo--------")
for query in test_queries:
    answer = find_best_answer(query)
    print(f"USer asked: {query}, BOt:{answer}\n")
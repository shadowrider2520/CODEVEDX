import re
import string

sample_queries = [
    "Hey how do i reset my password",
    "whats the wifi password for office",
    "Can i get the leave approved for tommorow",
    "IT ticket status pls check",
    "enaku odambu mudila edha help panen"
]

def normalize_text(text):
    return text.lower() # query and intent checking 

def remove_punctuation(text):
    return text.translate(str.maketrans("","",string.punctuation)) 
# str.maketrans create a translation rule map 
# with 2 arguments "" - char u want to replace, "" what to replace with ,string.punctuation-deletable chars

def tokenize(text):
    return text.split()

abbreviation_map = {
    "pwd":"password",
    "pls":"please",
    "whats":"what is",
    "u":"you",
    "idc":"i dont care",
    "wbu":"what about you",
    "fyn":"fine"
}

def expand_abbreviations(tokens):
    return [abbreviation_map.get(word,word) for word in tokens]

def preprocess(text):
    text = normalize_text(text)
    text = remove_punctuation(text)
    tokens = tokenize(text)
    tokens = expand_abbreviations(tokens)
    return " ".join(tokens)

print("||-==-=-=-==-=--==---=-=---NLP PREPROCESSING FOR CHATBOT-=-=-=-=-=-=-=-=-==--==-=--=||\n")
for query in sample_queries:
    cleaned = preprocess(query) # preprocess or clean the data firsst 
    print("Original:",query)
    print(f"Cleaned:{cleaned}\n")

# intents match our simple actions to their respective implemented functions and operations which can be dont
intent_keywords = {
    "password":"password_reset",
    "wifi":"wifi_access",
    "leave":"leave_request","ticket":"ticket_status"
}

def guess_intent(text):
    for keyword,intent  in intent_keywords.items():
        if keyword in text:
            return intent
        return "Invalid prompt"

print("\nBAsic intent guessing--.\n")
for query in sample_queries:
    cleaned = preprocess(query)
    intent = guess_intent(cleaned)
    print(f"Query -:> {query} , Cleaned -:> {cleaned} , intent: -:> {intent}")
    
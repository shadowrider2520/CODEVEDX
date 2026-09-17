import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
data = {
    "text": [
        "Scientists discover new exoplanet with signs of water",         # real
        "Doctors hate her! This one weird trick cures everything",        # fake
        "Government announces new infrastructure investment plan",         # real
        "Secret alien base found under the ocean, officials deny it",       # fake
        "Stock markets rally after positive economic data release",          # real
        "You won't believe what celebrities are hiding from you",             # fake
        "Local university receives grant for renewable energy research",       # real
        "Miracle pill melts fat overnight, doctors are furious",                 # fake
        "New public transport line opens to reduce city congestion",              # real
        "Breaking: moon landing was staged, new evidence revealed",                # fake
    ],
    "label": ["real","fake","real","fake","real","fake","real","fake","real","fake"]
}
dataset = pd.DataFrame(data)
dataset.to_csv("More_news.csv",index = False)
dataset  = pd.read_csv("More_news.csv")
print("Dataset loaded\n")
tfidf = TfidfVectorizer(stop_words="english",lowercase=True,max_features=100)
X = tfidf.fit_transform(dataset["text"]).toarray()
y = dataset["label"].values
print("Feature matrix shape:",X.shape)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print("Training samples:",X_train.shape[0]) # how many articles the ml model learns from
print("Test samples:",X_test.shape)
model = MultinomialNB()
model.fit(X_train,y_train)
y_pred = model.predict(X_test) # test the predictions
print("Predicted Labels:",y_pred)
print("Actual labels:",y_test)
accuracy = accuracy_score(y_test,y_pred)
print("\nBaseline model accuracy:",accuracy)
import joblib
joblib.dump(model,"fake_new_model.pkl")
joblib.dump(tfidf,"tdidf_vectorizer.pkl")
print("\nSaved - > fake_news_model.pkl,tfidf_vectorizer.pkl")
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

# ---- Step 1: Create the dataset ----
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
    "label": ["real", "fake", "real", "fake", "real", "fake", "real", "fake", "real", "fake"]
}

dataset = pd.DataFrame(data)

# ---- Save the dataset ----
dataset.to_csv("news_dataset.csv", index=False)
print("Dataset saved to More_news.csv")

# Reload it (simulating a real pipeline)
dataset = pd.read_csv("More_news.csv")
print("Dataset loaded:\n", dataset)

# ---- Step 2: Vectorize the text ----
tfidf = TfidfVectorizer(stop_words="english", lowercase=True, max_features=100)
X = tfidf.fit_transform(dataset["text"]).toarray()
y = dataset["label"].values
print("\nFeature matrix shape:", X.shape)

# ---- Step 3: Split into train/test ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Training samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])

# ---- Step 4: Train the model ----
model = MultinomialNB()
model.fit(X_train, y_train)

# ---- Step 5: Evaluate ----
y_pred = model.predict(X_test)
print("\nPredicted labels:", y_pred)
print("Actual labels:   ", y_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nBaseline model accuracy: {accuracy:.2f}")

# ---- Step 6: Save model + vectorizer ----
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
print("\nSaved → fake_news_model.pkl, tfidf_vectorizer.pkl")
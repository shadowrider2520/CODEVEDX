import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# Step 1: Load in some sample news
# (In a real pipeline this would come from a news dataset/API)
data = {
    "text": [
        "Breaking: Scientists confirm water found on Mars!",
        "You won't believe this ONE trick doctors HATE!!!",
        "Local government announces new budget for education.",
        "Aliens spotted in downtown, government covers it up!!!",
        "Stock market sees steady growth amid economic recovery."
    ],
    "label": ["real", "fake", "real", "fake", "real"]   # what we're trying to predict
}
dataset = pd.DataFrame(data)
dataset.to_csv("news_raw.csv", index=False)
dataset = pd.read_csv("news_raw.csv")

print("Raw articles loaded:\n", dataset)

# fidfVectorizer(stop_words="english") does cleaning + tokenizing + vectorizing together
# quick preview: what stopwords actually get removed
sample_vectorizer = TfidfVectorizer(stop_words="english")
sample_vectorizer.fit(["This is an example of the filtering"])
print("\nExample of words TfidfVectorizer treats as noise (stopwords):",
      sample_vectorizer.get_stop_words() and "the, is, a, an, of... (built-in English list)")

# Words that show up a lot in ONE
# article but rarely across the others get a higher score — those
# are usually the words that hint whether news is real or fake.

tfidf = TfidfVectorizer(stop_words="english", lowercase=True, max_features=100) # stepwords : remove default english propositions which arent required for word counting
# It consider first 100 words 
X = tfidf.fit_transform(dataset["text"]).toarray()   # cleaning + vectorizing happens here
y = dataset["label"].values # predicting values "Target"

print("\nVocabulary the model learned from our articles:\n", tfidf.get_feature_names_out()) # All the extracted words
print("\nTF-IDF matrix shape:", X.shape, "-> (num_articles, num_words)") 

# This shows the words appeared more times in the first artivcle
first_article_words = pd.DataFrame({
    "word": tfidf.get_feature_names_out(), # for that one of the words...
    "importance": X[0]  # the score of that word tf-idf score..
})
first_article_words = first_article_words[first_article_words["importance"] > 0].sort_values("importance", ascending=False)
# first article words gets updated with the words which only appear in the article 1 , hence sorted by importance or score it holds
print("\nMost important words in Article 1:\n", first_article_words)
# ---- Save the processed data so Day 15 can use it for model training ----
np.save("X_tfidf.npy", X)
np.save("y_labels.npy", y)
print("\nSaved -> X_tfidf.npy, y_labels.npy (ready for model training tomorrow)")
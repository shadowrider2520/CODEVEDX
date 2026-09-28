import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer # item descriptions to numeric vectors
from sklearn.metrics.pairwise import cosine_similarity


data = {
    "title": ["Inception", "Interstellar", "The Notebook", "Titanic", "The Matrix", "La La Land"],
    "description": [
        "sci-fi thriller dream heist mind bending action",
        "sci-fi space exploration time science emotional",
        "romance love story drama emotional relationship",
        "romance drama ship love tragedy historical",
        "sci-fi action hacker simulation reality mind bending",
        "romance music drama love dreams musical",
    ],
}
movies = pd.DataFrame(data)
tfidf = TfidfVectorizer(stop_words="english")
item_vectors = tfidf.fit_transform(movies["description"])
item_similarity = cosine_similarity(item_vectors)
similar_df = pd.DataFrame(item_similarity,index=movies["title"],columns=movies["title"])

def recommend_similar(liked_title,top_na=2):

    scores = similar_df[liked_title].drop(liked_title)
    return scores.sort_values(ascending=False).head(top_na) # most similar movie comes first

print("Because u liked Matrix....:)\n",recommend_similar("The Matrix"))
print("\nBecause you liked Titanic...:\n",recommend_similar("Titanic"))
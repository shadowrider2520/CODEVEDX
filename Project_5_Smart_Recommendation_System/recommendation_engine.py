import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import euclidean


# ============================================================
# 1. Sample dataset (users × products)
# ============================================================
dataset = pd.DataFrame({
    "Product_A": [5, 4, 0, 1],
    "Product_B": [3, 0, 4, 5],
    "Product_C": [4, 5, 0, 0],
    "Product_D": [0, 1, 5, 4],
}, index=["Alice", "Bob", "Carol", "Dave"])

print("=" * 50)
print("Original User-Product Ratings")
print("=" * 50)
print(dataset)


# ============================================================
# 2. Handle sparsity (zeros = "not rated", not "rated zero")
# ============================================================
def normalize_sparse_data(df):
    """Replace zeros with the mean of non-zero ratings for each product."""
    filled = df.copy()
    for column in filled.columns:
        non_zero_mean = filled[column][filled[column] > 0].mean()
        filled[column] = filled[column].replace(0, non_zero_mean)
    return filled


normalized_data = normalize_sparse_data(dataset)
print("\n" + "=" * 50)
print("Normalized Dataset (zeros → column means)")
print("=" * 50)
print(normalized_data.round(2))


# ============================================================
# 3. Similarity cache (avoid recomputing every time)
# ============================================================
class SimilarityCache:
    def __init__(self, data):
        self.dataset = data
        self._similarity_matrix = None

    def get_similarity_matrix(self):
        if self._similarity_matrix is None:
            print("\nComputing similarity matrix (not cached yet)...")
            matrix = cosine_similarity(self.dataset.values)
            self._similarity_matrix = pd.DataFrame(
                matrix, index=self.dataset.index, columns=self.dataset.index
            )
        else:
            print("Using cached similarity matrix.")
        return self._similarity_matrix

    def invalidate_cache(self):
        """Call this whenever the dataset changes (new user added, etc.)."""
        self._similarity_matrix = None


cache = SimilarityCache(normalized_data)
similarity_df = cache.get_similarity_matrix()

print("\n" + "=" * 50)
print("User Similarity Matrix (Cosine)")
print("=" * 50)
print(similarity_df.round(3))


# ============================================================
# 4. Euclidean distance example (alternative similarity)
# ============================================================
distance = euclidean(normalized_data.loc["Alice"], normalized_data.loc["Bob"])
print(f"\nEuclidean distance between Alice and Bob: {distance:.3f}")
print("(Smaller distance = more similar tastes)")


# ============================================================
# 5. Collaborative Filtering – recommend products
# ============================================================
def recommend_products(target_user, top_n=2):
    """Suggest products based on similar users' ratings."""
    similar_users = similarity_df[target_user].drop(target_user)
    similar_users = similar_users.sort_values(ascending=False)

    already_rated = dataset.loc[target_user]          # use original (with zeros)
    already_rated_products = already_rated[already_rated > 0].index.tolist()

    scores = pd.Series(dtype=float)
    for other_user, sim_score in similar_users.items():
        other_ratings = dataset.loc[other_user]
        weighted = other_ratings * sim_score
        scores = scores.add(weighted, fill_value=0)

    scores = scores.drop(already_rated_products, errors="ignore")
    return scores.sort_values(ascending=False).head(top_n)


print("\n" + "=" * 50)
print("Collaborative Recommendations")
print("=" * 50)
for user in dataset.index:
    print(f"\nFor {user}:")
    print(recommend_products(user))


# ============================================================
# 6. Content-based recommendations (item descriptions)
# ============================================================
movies = pd.DataFrame({
    "title": [
        "Inception", "Interstellar", "The Notebook",
        "Titanic", "The Matrix", "La La Land"
    ],
    "description": [
        "sci-fi thriller dream heist mind bending action",
        "sci-fi space exploration time science emotional",
        "romance love story drama emotional relationship",
        "romance drama ship love tragedy historical",
        "sci-fi action hacker simulation reality mind bending",
        "romance music drama love dreams musical",
    ],
})

tfidf = TfidfVectorizer(stop_words="english")
item_vectors = tfidf.fit_transform(movies["description"])
item_similarity = cosine_similarity(item_vectors)
item_sim_df = pd.DataFrame(
    item_similarity, index=movies["title"], columns=movies["title"]
)


def recommend_similar_items(liked_title, top_n=2):
    """Recommend items based on description similarity."""
    scores = item_sim_df[liked_title].drop(liked_title)
    return scores.sort_values(ascending=False).head(top_n)


print("\n" + "=" * 50)
print("Content-Based Recommendations (Movies)")
print("=" * 50)
print("\nBecause you liked The Matrix:")
print(recommend_similar_items("The Matrix"))
print("\nBecause you liked Titanic:")
print(recommend_similar_items("Titanic"))


# ============================================================
# 7. Hybrid scoring (blend collaborative + content)
# ============================================================
def hybrid_score(collaborative_score, content_score, weight=0.6):
    """
    weight = how much trust goes to collaborative filtering.
    (1 - weight) goes to content-based signals.
    """
    return (weight * collaborative_score) + ((1 - weight) * content_score)


print("\n" + "=" * 50)
print("Hybrid Score Example")
print("=" * 50)
example_collab = 0.75
example_content = 0.55
final = hybrid_score(example_collab, example_content)
print(f"Collaborative: {example_collab}")
print(f"Content-based: {example_content}")
print(f"Hybrid (weight=0.6): {final:.2f}")


# ============================================================
# 8. Add a new user to the dataset
# ============================================================
def add_user(dataset, user_name, ratings_dict):
    """Add a new user with their product ratings."""
    new_row = pd.DataFrame([ratings_dict], index=[user_name])
    return pd.concat([dataset, new_row])


print("\n" + "=" * 50)
print("Adding a New User")
print("=" * 50)
new_ratings = {
    "Product_A": 2,
    "Product_B": 5,
    "Product_C": 0,
    "Product_D": 4,
}
updated_dataset = add_user(dataset, "Suruli", new_ratings)
print(updated_dataset)

# After adding a user you would normally call:
# cache.invalidate_cache()
# and then recompute similarity on the updated (and re-normalized) data.

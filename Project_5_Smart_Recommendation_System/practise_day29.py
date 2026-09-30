# import numpy for numeric operations
import numpy as np
# import pandas for data handling
import pandas as pd
# import cosine_similarity for comparing users/items
from sklearn.metrics.pairwise import cosine_similarity

# ================================================================
# SCENARIO: Our recommendation logic works, but as more users and
# products get added, recalculating similarity for everyone every
# single time gets slow and repeats a lot of the same work. Today
# we optimize by caching, normalizing sparse data, and combining
# collaborative + content-based scores (hybrid approach).
# ================================================================

# sample dataset (same structure as before)
dataset = pd.DataFrame({
    "Product_A": [5, 4, 0, 1],
    "Product_B": [3, 0, 4, 5],
    "Product_C": [4, 5, 0, 0],
    "Product_D": [0, 1, 5, 4],
}, index=["Alice", "Bob", "Carol", "Dave"])

# ---- Optimization 1: Handle sparsity properly ----
# 0 usually means "not rated", not "rated zero" — treating it as a
# real rating skews similarity. Replace 0s with each product's mean
# rating (a common trick called mean imputation) so missing data
# doesn't distort the similarity comparison.
def normalize_sparse_data(df):
    filled = df.copy()
    for column in filled.columns:
        # calculate the mean rating for this product, ignoring zeros
        non_zero_mean = filled[column][filled[column] > 0].mean()
        # replace zeros with that mean instead of leaving them as 0
        filled[column] = filled[column].replace(0, non_zero_mean)
    return filled

normalized_data = normalize_sparse_data(dataset)
print("Normalized dataset (zeros replaced with column means):\n", normalized_data.round(2))

# ---- Optimization 2: Cache the similarity matrix instead of recomputing ----
class SimilarityCache:
    def __init__(self, dataset):
        self.dataset = dataset
        self._similarity_matrix = None    # cache storage, empty at first

    def get_similarity_matrix(self):
        # only recompute if we haven't cached it yet
        if self._similarity_matrix is None:
            print("Computing similarity matrix (not cached yet)...")
            matrix = cosine_similarity(self.dataset.values)
            self._similarity_matrix = pd.DataFrame(
                matrix, index=self.dataset.index, columns=self.dataset.index
            )
        else:
            print("Using cached similarity matrix.")
        return self._similarity_matrix

    def invalidate_cache(self):
        # call this whenever the dataset changes (new user added, etc.)
        self._similarity_matrix = None

cache = SimilarityCache(normalized_data)
print("\nFirst call:")
sim1 = cache.get_similarity_matrix()   # computes fresh
print("\nSecond call:")
sim2 = cache.get_similarity_matrix()   # uses cache, much faster

# ---- Optimization 3: Hybrid scoring (blend collaborative + content signals) ----
def hybrid_score(collaborative_score, content_score, weight=0.6):
    # weight controls how much trust goes to collaborative vs content signals
    return (weight * collaborative_score) + ((1 - weight) * content_score)

# example: blending scores from Day 26 (collaborative) and Day 27 (content-based)
example_collab_score = 0.75
example_content_score = 0.55
final_score = hybrid_score(example_collab_score, example_content_score)
print(f"\nHybrid recommendation score: {final_score:.2f}")
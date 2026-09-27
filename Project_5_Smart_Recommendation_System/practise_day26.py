import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

data = {
    "user":["Mithun","Navina","gomathi","darshan"],
    "Product_A": [5, 4, 0, 1],
    "Product_B": [3, 0, 4, 5],
    "Product_C": [4, 5, 0, 0],
    "Product_D": [0, 1, 5, 4],
}
# rows are users columns are products
dataset = pd.DataFrame(data)
# user as index so we take reference rows by name
dataset = dataset.set_index("user")
similarity_matrix = cosine_similarity(dataset.values)
similarity_df = pd.DataFrame(similarity_matrix,index=dataset.index,columns = dataset.index)

# function recommends products
def recommend_products(target_user,top_n=2):
    # get similarity scores bw target user and everyone
    similar_users = similarity_df[target_user].drop(target_user)
    similar_users = similar_users.sort_values(ascending=False)

     # get the products target user already rated
    already_rated = dataset.loc[target_user]
    already_rated_products = already_rated[already_rated>0].index.tolist()

    scores = pd.Series(dtype=float)
    for other_user, similarity_score in similar_users.items():
        # get that other user's ratings
        other_ratings = dataset.loc[other_user]
        # weight their ratings by how similar they are to our target user
        weighted_ratings = other_ratings * similarity_score
        # add these weighted scores to our running total
        scores = scores.add(weighted_ratings, fill_value=0)
    # remove products the target user already tried
    scores = scores.drop(already_rated_products,errors="ignore")
    # sort remaining candidate products by weighted score the highest first
    recommendations = scores.sort_values(ascending=False).head(top_n)
    return recommendations

for user in dataset.index:
    print(f"\nRecommendations for {user}")
    print(recommend_products(user))
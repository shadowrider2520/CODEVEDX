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

# row = users , column = products
dataset = pd.DataFrame(data)
print("User-Product ratings:\n",dataset)

# extract just the numeric rating columns(drop the "user name column")
ratings_matrix = dataset.drop("user",axis=1).values

# cosine similarity bw every pair of users        
user_similarity = cosine_similarity(ratings_matrix)
# wrap the result in a DataFrame with user names as labels for readability
similarity_df = pd.DataFrame(user_similarity,index=dataset["user"],columns= dataset["user"])
# print the matrix
print("\nUIser Similarity matrix:\n",similarity_df)

# which user is most similar to gomathi
gomathi_similarities =  similarity_df["gomathi"].drop("gomathi")
# sort the simolarities from high to low
most_similar_user = gomathi_similarities.sort_values(ascending=False).index[0]
print(f"\nUser most similar to gomathi:{most_similar_user}")

# Euclidean distance as an alternative similarity measure
from scipy.spatial.distance import euclidean

# straight line distance between Mithun and Navina rating vectors 
distance_mithun_navina = euclidean(ratings_matrix[0],ratings_matrix[1])

# smaller distance = more similar taste
print(f"\nEuclidean distance between mithun and navina:{distance_mithun_navina}")
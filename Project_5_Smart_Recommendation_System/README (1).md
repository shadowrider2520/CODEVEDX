# Smart Recommendation System

A simple recommendation engine that suggests products (or movies) based on what similar users liked and what the items themselves are about.

Built step-by-step over a few practice days — starting from basic user similarity all the way to a hybrid approach that mixes collaborative filtering with content-based signals.

---

## What it can do

- **User similarity** – Finds which users have the most similar tastes using cosine similarity (and also shows Euclidean distance as an alternative).
- **Collaborative recommendations** – Looks at users similar to you and suggests products they rated highly that you haven’t tried yet.
- **Content-based recommendations** – Recommends items based on their descriptions (TF-IDF + cosine similarity), so it works even when rating data is thin.
- **Add new users** – Lets you collect ratings from a new user and fold them into the existing dataset.
- **Sparsity handling** – Replaces missing ratings (the zeros) with product means so the similarity calculations don’t get skewed.
- **Caching** – Keeps the similarity matrix in memory so you don’t recompute it every single time.
- **Hybrid scoring** – Blends collaborative and content-based scores with a simple weight so you can tune how much you trust each signal.

---

## How to run it

1. Install the dependencies:

```bash
pip install pandas numpy scikit-learn scipy
```

2. Run the main file:

```bash
python recommendation_engine.py
```

That’s it. The script will print the normalized ratings, similarity matrix, and a few example recommendations.

---

## Project structure

```
Project_5_Smart_Recommendation_System/
├── recommendation_engine.py   # Final combined logic (similarity, recommendations, caching, hybrid)
├── practise_day_25.py         # User similarity (cosine + Euclidean)
├── practise_day26.py          # Collaborative filtering recommendations
├── practise_day27.py          # Content-based movie recommendations
├── practise_day28.py          # Adding a new user + collecting ratings
├── practise_day29.py          # Sparsity handling, caching, hybrid scoring
└── README.md                  # You are here
```

---

## Quick notes

- The sample data is tiny on purpose so you can see exactly what’s happening.
- Zeros mean “not rated,” not “rated zero.” The normalization step takes care of that.
- You can change the hybrid weight (default 0.6 collaborative / 0.4 content) depending on how much rating data you have.

Feel free to expand the catalog, add more users, or swap in real product descriptions. The core pieces are already there.

---

# 🎬 Content-Based Filtering Movie Recommender System

This project implements a **custom neural-network-based content-based recommender system** trained on the MovieLens dataset.
It predicts user ratings and learns meaningful **movie embedding vectors** that can later be used for **movie–movie similarity search**.

---

## 📊 Dataset

The dataset originates from the **MovieLens “ml-latest-small”** collection.

**Reference:**
F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context*. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19.

### Original Dataset

* **610 users**
* **9,724 movies**
* Ratings from **0.5 to 5.0** in increments of 0.5

---

## 🧹 Data Preprocessing & Feature Engineering

Several steps were applied to clean and enrich the dataset:

### ✔ Cleaning

* Removed **all movies without at least one genre**.
* Removed movies with **missing release year**.
* Kept only movies released **from the year 2000 onward**.
* Removed users and movies that had **only one rating**.

### ✔ Feature Engineering

* Computed **per-user average rating**.
* Computed **per-user average rating per genre**.
* Computed **average rating per movie**.
* Applied **one-hot encoding** to movie genres.
* Created **user feature vectors** and **movie feature vectors** for the neural network.

### Final Dataset Size

* **443 users**
* **4,764 movies**

---

## 🤖 Model Architecture

The model consists of **two sub-models**:

### 1️⃣ User Sub-model

Processes user-level features (user averages, behaviors, etc.).

### 2️⃣ Movie Sub-model

Processes movie-level features (genres, year, average rating, etc.).

These two branches are merged using the **Keras Functional API**, and the combined vector is passed through dense layers to produce a **predicted rating**.

---

## 📈 Evaluation

Performance metrics (Mean Squared Error & RMSE):

| Dataset | MSE  | RMSE |
| ------- | ---- | ---- |
| Train   | 0.41 | 0.64 |
| CV      | 0.60 | 0.77 |
| Test    | 0.60 | 0.77 |

---

## 🔍 Embedding Model for Movie Similarity

A second model (`model_m`) is trained to output **movie embedding vectors** ( v_m ).

Using these embedding vectors:

* A **movie–movie distance matrix** is constructed using squared Euclidean distance
  [
  \text{dist}[i,j] = | v_m^{(i)} - v_m^{(j)} |^2.
  ]
* This matrix is then used for **content-based similarity search**, allowing us to retrieve the most similar movies to any given movie.

We also:

* Predict movies for a **new user** to confirm the model recommends films with genres similar to what the user liked.
* Predict for an **existing user** and compare predicted vs. actual ratings.

---

## 📂 Project Files

| File                          | Description                                                        |
| ----------------------------- | ------------------------------------------------------------------ |
| `cleaned_ratings.csv`         | Processed ratings data                                             |
| `cleaned_movies.csv`          | Processed movies data                                              |
| `movies.csv`                  | Movies after removing those without genres                         |
| `ratings.csv`                 | Ratings aligned with the filtered movie list                       |
| `movies_features.csv`         | Movie feature matrix                                               |
| `user_features.csv`           | User feature matrix                                                |
| `X_users.csv` / `X_u.npy`     | User input dataset (before splitting)                              |
| `X_movies.csv` / `X_m.npy`    | Movie input dataset (before splitting)                             |
| `Y.npy`                       | Ratings vector (**Y[i] = rating of user X_u[i] for movie X_m[i]**) |
| Train / CV / Test CSVs        | Dataset splits                                                     |


---



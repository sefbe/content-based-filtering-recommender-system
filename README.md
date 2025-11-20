# 🎬 Collaborative Filtering Movie Recommender System

This project implements a **custom collaborative filtering recommender system** using **matrix factorization**. The goal is to predict user ratings and recommend movies released from the year 2000 onward.

---

## 📊 Dataset

The dataset used in this project is derived from the **MovieLens "ml-latest-small" dataset**.

**Reference:**
F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context*. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. [https://doi.org/10.1145/2827872](https://doi.org/10.1145/2827872)

### Original dataset statistics

* **610 users**
* **9724 movies**
* Ratings on a scale from 0.5 to 5.0 in 0.5 increments

### Processed dataset

The dataset has been filtered and reduced as follows:

* Only movies released **from the year 2000 onward** are included.
* Movies without a release year have been removed.
* Users and movies with only a single rating (especially if their only rating falls in the test or CV set) have been removed.

**Final dataset statistics:**

* **443 users**
* **4375 movies**

---

## 🔹 Model Overview

The recommender system is implemented **from scratch** using **matrix factorization**. TensorFlow is used **only for tensor operations and gradient optimization**.

* Trainable parameters include user and movie latent factors and user bias.
* Model is trained with **gradient descent** and **regularization**.
* Evaluated using **Mean Squared Error (MSE)** on training, cross-validation, and test sets.

### Performance

| Dataset | MSE   | RMSE |
| ------- | ------|------|
| Train   | 0.55  | 0.74 |
| CV      | 0.75  | 0.86 |
| Test    | 0.76  | 0.87 |

### Features

* Personalized movie recommendations
* Comparison of predicted vs actual ratings
* Recommendations filtered for movies released after 2000

---

## 📂 Project Files

* `cleaned_ratings.csv` — cleaned ratings data
* `cleaned_movies.csv` — cleaned movie data
* `Y.npy` / `R.npy` — rating and indicator matrices for training
* Train / CV / Test splits saved as CSVs
* Training and prediction scripts

---




import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from pathlib import Path

class KMeans_Robusto:
    def __init__(self, n_clusters=10, max_iter=300, tol=1e-4, n_init=10, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None

    def _init_centroids_kmeanspp(self, X):
        n_samples, _ = X.shape
        centroids = np.empty((self.n_clusters, X.shape[1]), dtype=float)
        rng = np.random.RandomState(self.random_state)
        centroids[0] = X[rng.randint(0, n_samples)]

        for k in range(1, self.n_clusters):
            dist_sq = np.min(np.square(np.linalg.norm(X[:, None, :] - centroids[None, :k, :], axis=2)), axis=1)
            probs = dist_sq / np.sum(dist_sq)
            cumulative_probs = np.cumsum(probs)
            r = rng.rand()
            idx = np.searchsorted(cumulative_probs, r)
            centroids[k] = X[idx]
        return centroids

    def _assign_clusters(self, X, centroids):
        distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            points = X[labels == k]
            if len(points) > 0:
                new_centroids[k] = np.mean(points, axis=0)
            else:
                new_centroids[k] = X[np.random.randint(0, X.shape[0])]
        return new_centroids

    def _compute_inertia(self, X, centroids, labels):
        inertia = 0.0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            inertia += np.sum((cluster_points - centroids[k]) ** 2)
        return inertia

    def fit(self, X):
        X = np.asarray(X)
        rng = np.random.RandomState(self.random_state)
        best_inertia = np.inf
        best_centroids = None
        best_labels = None

        for _ in range(self.n_init):
            centroids = self._init_centroids_kmeanspp(X)
            for _ in range(self.max_iter):
                labels = self._assign_clusters(X, centroids)
                new_centroids = self._update_centroids(X, labels)
                shift = np.linalg.norm(new_centroids - centroids)
                centroids = new_centroids
                if shift < self.tol:
                    break
            inertia = self._compute_inertia(X, centroids, labels)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels

        self.centroids = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        return self

    def predict(self, X):
        X = np.asarray(X)
        return self._assign_clusters(X, self.centroids)

    def save(self, filename):
        data = {
            "n_clusters": self.n_clusters,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "n_init": self.n_init,
            "random_state": self.random_state,
            "centroids": self.centroids,
            "inertia_": self.inertia_
        }
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        print(f"Modelo guardado en: {filename}")

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        model = KMeans_Robusto(
            n_clusters=data["n_clusters"],
            max_iter=data["max_iter"],
            tol=data["tol"],
            n_init=data["n_init"],
            random_state=data["random_state"]
        )
        model.centroids = data["centroids"]
        model.inertia_ = data["inertia_"]
        print(f"Modelo cargado desde: {filename}")
        return model

# ============================================================
# EVALUACIÓN MASIVA DE UMAPs
# ============================================================

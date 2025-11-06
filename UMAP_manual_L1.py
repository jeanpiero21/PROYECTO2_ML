import numpy as np
import pickle
import os
from sklearn.neighbors import NearestNeighbors

class UMAP_manual_L1:
    def __init__(self, n_neighbors=15, n_components=20, n_epochs=50, learning_rate=1.0, random_state=42):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.indices_ = None
        self.weights_ = None
        self.embedding_ = None

    def fit(self, X):
        np.random.seed(self.random_state)
        n_samples = X.shape[0]

        # k-vecinos más cercanos con distancia Manhattan ---
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, metric='manhattan').fit(X)
        distances, indices = nbrs.kneighbors(X)

        # Convertir distancias a pesos (kernel exponencial) ---
        sigma = np.mean(distances[:, -1])
        weights = np.exp(-distances / (2 * sigma))   # usar distancia L1, sin elevar al cuadrado

        # Inicialización aleatoria ---
        X_UMAP = np.random.randn(n_samples, self.n_components) * 0.01

        # Optimización vectorizada ---
        for epoch in range(self.n_epochs):
            for i in range(n_samples):
                neighbors = indices[i]
                diffs = X_UMAP[i] - X_UMAP[neighbors]          # (n_neighbors, n_components)
                
                # Distancia Manhattan entre embeddings
                dists = np.sum(np.abs(diffs), axis=1, keepdims=True) + 1e-8

                # Gradiente según distancia L1
                direction = np.sign(diffs)                     # dirección del gradiente (signo)
                w = weights[i][:, None]
                grads = w * direction / (1 + dists)            # gradiente suavizado

                # Actualización
                X_UMAP[i] -= self.learning_rate * np.sum(grads, axis=0)
                X_UMAP[neighbors] += self.learning_rate * grads

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.n_epochs} completada")

        self.embedding_ = X_UMAP
        self.indices_ = indices
        self.weights_ = weights
        return self

    def transform(self, X):
        if self.embedding_ is None:
            raise ValueError("Primero debes llamar a fit() antes de transform().")
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, metric='manhattan').fit(X)
        distances, indices = nbrs.kneighbors(X)
        sigma = np.mean(distances[:, -1])
        weights = np.exp(-distances / (2 * sigma))
        X_new = np.random.randn(X.shape[0], self.n_components) * 0.01
        for epoch in range(10):
            for i in range(X.shape[0]):
                neighbors = indices[i]
                diffs = X_new[i] - self.embedding_[neighbors]
                dists = np.sum(np.abs(diffs), axis=1, keepdims=True) + 1e-8
                direction = np.sign(diffs)
                w = weights[i][:, None]
                grads = w * direction / (1 + dists)
                X_new[i] -= self.learning_rate * np.sum(grads, axis=0)
                X_new[neighbors] += self.learning_rate * grads
        return X_new

    def fit_transform(self, X):
        self.fit(X)
        return self.embedding_

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "n_neighbors": self.n_neighbors,
                "n_components": self.n_components,
                "n_epochs": self.n_epochs,
                "learning_rate": self.learning_rate,
                "random_state": self.random_state,
                "embedding_": self.embedding_,
                "indices_": self.indices_,
                "weights_": self.weights_
            }, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = UMAP_manual_L1(
            n_neighbors=data["n_neighbors"],
            n_components=data["n_components"],
            n_epochs=data["n_epochs"],
            learning_rate=data["learning_rate"],
            random_state=data["random_state"]
        )
        obj.embedding_ = data["embedding_"]
        obj.indices_ = data["indices_"]
        obj.weights_ = data["weights_"]
        return obj

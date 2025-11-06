import numpy as np
import pickle
import os
from scipy.sparse.linalg import svds

class PCA_manual:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None

    def fit(self, X):
        # Centrar los datos
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Calcular SVD truncado
        U, S, Vt = svds(X_centered, k=self.n_components)

        # Seleccionar n_components
        self.components_ = Vt[:self.n_components, :]

        # Varianza explicada
        self.explained_variance_ = (S[:self.n_components] ** 2) / (X.shape[0] - 1)

        return self

    def transform(self, X):
        # Aplicar la transformación PCA usando los componentes entrenados
        if self.mean_ is None or self.components_ is None:
            raise ValueError("Primero debes llamar a fit() antes de transform().")
        X_centered = X - self.mean_
        X_pca = np.dot(X_centered, self.components_.T)
        return X_pca

    def fit_transform(self, X):
        # Entrenar y aplicar transformación en una sola llamada
        self.fit(X)
        return self.transform(X)

    def save(self, path):
        # Guardar el modelo entrenado en un archivo .pkl
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "n_components": self.n_components,
                "mean_": self.mean_,
                "components_": self.components_,
                "explained_variance_": self.explained_variance_
            }, f)

    @staticmethod
    def load(path):
        # Cargar un modelo PCA previamente guardado
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = PCA_manual(data["n_components"])
        obj.mean_ = data["mean_"]
        obj.components_ = data["components_"]
        obj.explained_variance_ = data["explained_variance_"]
        return obj

import numpy as np
import graphtools as gt  # Assuming 'graphtools' is a valid package

class JLMA:
    def __init__(self, k=5, decay=40, mu=1, d=2, normalized_laplacian=True):
        self.k = k
        self.decay = decay
        self.mu = mu
        self.d = d
        self.normalized_laplacian = normalized_laplacian

    def construct_affinities(self, X):
        graph = gt.Graph(X, knn=self.k, decay=self.decay)
        W = graph.K.toarray()
        return W

    def laplacian_from_affinities(self, W):
        D = np.diag(W.sum(axis=1))
        L = D - W

        if self.normalized_laplacian:
            D_inv_sqrt = np.diag(1.0 / np.sqrt(W.sum(axis=1)))
            L_norm = D_inv_sqrt @ L @ D_inv_sqrt
            return L_norm

        return L

    def get_joint_laplacian(self, X1, X2, correspondences):
        n1, _ = X1.shape
        n2, _ = X2.shape

        W = np.zeros((n1 + n2, n1 + n2))

        W1 = self.construct_affinities(X1)
        W2 = self.construct_affinities(X2)

        W[:n1, :n1] = W1
        W[n1:, n1:] = W2


        self.W1 = W1
        self.W2 = W2


        for (i, j) in correspondences:
            W[i, n1 + j] = 1
            W[n1 + j, i] = 1

        self.W = W
        L = self.laplacian_from_affinities(W)

        return L

    def fit(self, X1, X2, correspondences):
        n1, m1 = X1.shape
        n2, m2 = X2.shape

        M = np.zeros((n1 + n2, n1 + n2))
        M[:n1, :n1] = np.eye(n1)
        M[n1:, n1:] = np.eye(n2)

        X = np.vstack((X1, X2))

        self.X1 = X1
        self.X2 = X2
        self.X  = X

        L = self.get_joint_laplacian(X1, X2, correspondences)
        self.L = L

        A = X.T @ L @ X
        B = X.T @ (M + self.mu * np.eye(n1 + n2)) @ X

        eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(B) @ A)
        idx = np.argsort(eigvals)
        eigvecs = eigvecs[:, idx]

        Y = X @ eigvecs[:, :self.d]
        Y1 = Y[:n1, :]
        Y2 = Y[n1:, :]

        self.Y = Y
        self.Y1 = Y1
        self.Y2 = Y2
        self.eigvecs = eigvecs

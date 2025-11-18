import numpy as np


class NormalEquationRegressor:
    """Simple linear regressor using the normal equation (closed-form).


    This implementation does not include regularization. For numerical stability,
    a small ridge (lambda * I) might be added if X^T X is singular.
    """
    def __init__(self, add_bias=True, ridge_lambda=0.0):
        self.add_bias = add_bias
        self.ridge_lambda = ridge_lambda
        self.theta = None


    def fit(self, X, y):
        # X: (n_samples, n_features)
        X_mat = np.array(X)
        y_vec = np.array(y).reshape(-1, 1)
        if self.add_bias:
            X_mat = np.hstack([np.ones((X_mat.shape[0], 1)), X_mat])


    # Normal equation with optional ridge
        A = X_mat.T.dot(X_mat)
        if self.ridge_lambda > 0:
            A = A + self.ridge_lambda * np.eye(A.shape[0])
        self.theta = np.linalg.pinv(A).dot(X_mat.T).dot(y_vec)
    # store as 1D
        self.theta = self.theta.ravel()
        return self


    def predict(self, X):
        X_mat = np.array(X)
        if self.add_bias:
            X_mat = np.hstack([np.ones((X_mat.shape[0], 1)), X_mat])
        return X_mat.dot(self.theta)
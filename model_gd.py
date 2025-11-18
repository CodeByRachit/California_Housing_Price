import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionGD:
    """
    Linear Regression using Gradient Descent
    ----------------------------------------
    Parameters:
        learning_rate : float
            Step size for gradient descent updates.
        n_iters : int
            Number of iterations for optimization.

    Attributes:
        weights : np.ndarray
            Model coefficients (shape: [n_features, 1]).
        bias : float
            Model bias term.
        losses : list
            Stores MSE loss values for each iteration.
    """

    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.losses = []  # Track loss at each iteration

    def fit(self, X, y, verbose=False):
        """
        Train the linear regression model using gradient descent.
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float).reshape(-1, 1)

        m, n = X.shape  # m = samples, n = features
        self.weights = np.zeros((n, 1))
        self.bias = 0

        for i in range(self.n_iters):
            # Predictions
            y_pred = np.dot(X, self.weights) + self.bias

            # Compute error and loss (MSE)
            error = y_pred - y
            loss = np.mean(error ** 2)
            self.losses.append(loss)

            # Compute gradients
            dw = (1/m) * np.dot(X.T, error)
            db = (1/m) * np.sum(error)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Print progress occasionally
            if verbose and i % (self.n_iters // 10) == 0:
                print(f"Iteration {i:4d} | Loss: {loss:.6f}")

    def predict(self, X):
        """
        Predict target values for given input X.
        """
        X = np.array(X, dtype=float)
        return np.dot(X, self.weights) + self.bias

    def plot_convergence(self):
        """
        Plot training loss (MSE) vs iterations.
        """
        if not self.losses:
            print("⚠️ No loss data found. Train the model first using fit().")
            return
        plt.figure(figsize=(7, 4))
        plt.plot(self.losses, label='Training Loss (MSE)')
        plt.xlabel('Iteration')
        plt.ylabel('Mean Squared Error')
        plt.title(f'Gradient Descent Convergence (lr={self.learning_rate})')
        plt.legend()
        plt.grid(True)
        plt.show()

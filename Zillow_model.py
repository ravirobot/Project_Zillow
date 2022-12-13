class LinearRegression:

    def predict(self, X):
        return np.dot(X, self._W)

    def _gradient_descent_step(self, X, targets, lr):
        predictions = self.predict(X)

        error = predictions - targets
        gradient = np.dot(X.T, error) / len(X)

        self._W -= lr * gradient

    def fit(self, X, y, n_iter=100000, lr=0.01):
        self._W = np.zeros(X.shape[1])

        for i in range(n_iter):
            self._gradient_descent_step(x, y, lr)

        return self
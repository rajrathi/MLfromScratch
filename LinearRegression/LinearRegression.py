import numpy as np

class LinearRegression():

    def __init__(self, lr = 0.01, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weigts = None
        self.bias = None


    def fit(self, X, y):
        n_samlples, n_features  = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in (self.n_iters):
            y_pred = np.dot(X.T, self.weights + self.bias)

            gradient_weight = 2 * np.dot(X.T, (y_pred - y)) / n_samlples
            gradient_bias = 2 * np.sum(y_pred - y) / n_samlples

            self.weights -= (self.lr *  gradient_weight)
            self.bias -= (self.lr * gradient_bias)
        
        
    def predict(self, X):
        return np.dot(X.T, self.weights) + self.bias

    
    def evaluate(self, y_pred, y):
        
        r2_score = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        mse = np.mean((y - y_pred) ** 2)
        rmse = np.sqrt(mse)

        print("Evaluation Summary")
        print("=======================")
        print(f"MSE:{mse}")
        print(f"RMSE:{rmse}")
        print(f"R-squared:{r2_score}")
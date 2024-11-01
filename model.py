import numpy as np

class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def initialize_parameters(self, n_features):
        self.weights = np.zeros(n_features)
        self.bias = 0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def train(self, X, y):
        n_samples, n_features = X.shape
        self.initialize_parameters(n_features)

        for i in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        return y_predicted


class LogisticRegressionOvA:
    def __init__(self, num_classes=10, learning_rate=0.01, n_iterations=1000):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.models = []

    def train(self, X, y):
        for i in range(self.num_classes):
            print(f"Training model for class {i}")
            binary_y = (y == i).astype(int) 
            model = LogisticRegressionScratch(self.learning_rate, self.n_iterations)
            model.train(X, binary_y)
            self.models.append(model)

    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        return np.argmax(predictions, axis=0)  

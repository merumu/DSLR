import pandas as pd
import numpy as np

class LogisticRegression:
    
    def __init__(self, alpha=0.001, max_iter=1000, verbose=False, learning_rate='constant'):
        self.alpha = alpha
        self.max_iter = max_iter
        self.verbose = verbose
        self.learning_rate = learning_rate # can be 'constant' or 'invscaling'
        self.thetas = []
        # Your code here (e.g. a list of loss for each epochs...)
    
    def fit(self, x_train, y_train):
        """
        Fit the model according to the given training data.
        Args:
            x_train: a 1d or 2d numpy ndarray for the samples
            y_train: a scalar or a numpy ndarray for the correct labels
        Returns:
        self : object
            None on any error.
        Raises:
            This method should not raise any Exception.
        """
        self.thetas = np.zeros(x_train.shape[1] + 1)
        x_new = np.insert(x_train, 0, 1, axis=1)
        y_true = np.array(y_train)
        for n in range(self.max_iter):
            y_pred = self.predict(x_train)
            grad = self._vec_log_gradient_(x_new, y_true, y_pred)
            self.thetas = self.thetas - self.alpha * (1/x_train.shape[0]) * grad
        print(self.thetas)
    
    def predict(self, x_train):
        """
        Predict class labels for samples in x_train.
        Arg:
            x_train: a 1d or 2d numpy ndarray for the samples
        Returns:
            y_pred, the predicted class label per sample.
            None on any error.
        Raises:
            This method should not raise any Exception.
        """
        if x_train.shape[1] == self.thetas.shape[0] - 1:
            X = np.insert(x_train, 0, 1, axis=1)
            return X.dot(self.thetas)
        return None
    
    def score(self, x_train, y_train):
        """
        Returns the mean accuracy on the given test data and labels.
        Arg:
            x_train: a 1d or 2d numpy ndarray for the samples
            y_train: a scalar or a numpy ndarray for the correct labels
        Returns:
            Mean accuracy of self.predict(x_train) with respect to y_true
            None on any error.
        Raises:
            This method should not raise any Exception.
        """
        y_pred = self.predict(x_train)
        y_true = np.array(y_train)
        return (y_pred == y_true).mean()

    def _sigmoid(self, x):
        if isinstance(x, list):
            ret = []
            for n in x:
                ret.append(1 / (1 + exp(-n)))
            return ret
        if isinstance(x, np.ndarray):
            return 1 / (1 + np.exp(-x))
        return 1 / (1 + exp(-x))

    def _vec_log_gradient_(self, x, y_true, y_pred):
        if isinstance(x, np.ndarray) and isinstance(y_pred, np.ndarray) and isinstance(y_true, np.ndarray):
            return (y_pred - y_true).dot(x)
        return (y_pred - y_true) * x

    def _vec_log_loss_(self, y_true, y_pred, m, eps=1e-15):
        if isinstance(y_pred, np.ndarray) and isinstance(y_true, np.ndarray):
            return (-1/m) * (y_true.dot(np.log(y_pred)) + (1 - y_true).dot(np.log(1 - y_pred)))
        return (-1/m) * (y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))

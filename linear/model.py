import numpy as np
from linear_regression.utils.logger import get_logger

logger = get_logger("linear_regression", "logs/linear_regression.log")


class SimpleLinearRegression:
    def __init__(self):
        self.m = None
        self.b = None
        logger.info("Initialized SimpleLinearRegression")

    def fit(self, X, y):
        try:
            logger.info("Starting training")

            X = np.array(X).flatten()
            y = np.array(y)

            mean_x = np.mean(X)
            mean_y = np.mean(y)

            numerator = np.sum((X - mean_x) * (y - mean_y))
            denominator = np.sum((X - mean_x) ** 2)

            if denominator == 0:
                raise ValueError("Division by zero in slope calculation")

            self.m = numerator / denominator
            self.b = mean_y - self.m * mean_x

            logger.info(f"Training completed: m={self.m}, b={self.b}")

        except Exception as e:
            logger.exception(f"Training failed: {e}")
            raise

    def predict(self, X):
        X = np.array(X).flatten()
        return self.m * X + self.b

    def get_params(self):
        return self.m, self.b
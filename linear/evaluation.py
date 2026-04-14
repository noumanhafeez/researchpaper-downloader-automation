import numpy as np


def evaluate_model(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)

    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)

    r2 = 1 - (ss_residual / ss_total)

    return {
        "mse": mse,
        "r2_score": r2
    }
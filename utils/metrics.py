import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error.
    Zero actual values are ignored.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask = y_true != 0

    if not np.any(mask):
        return np.nan

    return np.mean(
        np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    ) * 100


def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
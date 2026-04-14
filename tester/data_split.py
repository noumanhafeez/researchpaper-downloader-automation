import pandas as pd

from sklearn.model_selection import train_test_split
from ..utils.logger import get_logger

logger = get_logger("data_splitter", "logs/data_splitter.log")


def split_data(df: pd.DataFrame):
    """
    Splits a DataFrame into training and testing sets.
    Logs the process.

    Args:
        df (pd.DataFrame): The input dataset.
        target_col (str): Name of the target column.
        test_size (float): Fraction of data to use for testing.
        random_state (int): Random seed for reproducibility.

    Returns:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features
        y_train (pd.Series): Training target
        y_test (pd.Series): Testing target
    """
    try:
        logger.info(f"Starting data split. Training size: 80%. Test size: 20%, Random state: 42")

        X = df.drop(columns=['class'])
        y = df['class']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        logger.info(f"Data split completed. "
                    f"Training shape: X={X_train.shape}, y={y_train.shape}; "
                    f"Testing shape: X={X_test.shape}, y={y_test.shape}")

        return X_train, X_test, y_train, y_test

    except Exception as e:
        logger.error(f"Unexpected error during data split: {e}")
        raise
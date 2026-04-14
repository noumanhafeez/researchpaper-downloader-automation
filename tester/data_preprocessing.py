import pandas as pd
import sklearn
from decision_tree.src.data_ingestion import load_data_from_kaggle
from sklearn.preprocessing import LabelEncoder
from ..utils.logger import get_logger

logger = get_logger("data_loader", "logs/data_loader.log")


def preprocess() -> pd.DataFrame:
    try:
        logger.info("Starting preprocessing")

        # Load data
        df = load_data_from_kaggle("uciml/mushroom-classification")
        logger.info(f"Data loaded successfully. Shape: {df.shape}")

        # Select features
        select_features = df[['gill-size', 'odor', 'cap-surface', 'class']]
        logger.info(f"Selected features: {list(select_features.columns)}")

        new_df = select_features.copy()

        # Encoding
        logger.info("Applying Label Encoding")

        for col in select_features.columns:
            le = LabelEncoder()
            new_df[col] = le.fit_transform(df[col])
            logger.debug(f"Encoded column: {col}")

        logger.info(f"Preprocessing completed. Final shape: {new_df.shape}")

        return new_df

    except Exception as e:
        logger.exception(f"Error during preprocessing: {e}")
        raise
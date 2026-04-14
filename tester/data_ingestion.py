import pandas as pd
from pathlib import Path
import kagglehub

from decision_tree.utils.logger import get_logger

logger = get_logger("data_loader", "logs/data_loader.log")


def load_data_from_kaggle(dataset: str) -> pd.DataFrame:
    """
    Download dataset from Kaggle and return DataFrame
    """
    try:
        logger.info(f"Downloading dataset: {dataset}")

        # download dataset
        path = kagglehub.dataset_download(dataset)
        logger.info(f"Downloaded to: {path}")

        # find csv file
        folder = Path(path)
        csv_files = list(folder.glob("*.csv"))

        if not csv_files:
            logger.error("No CSV file found in dataset")
            raise FileNotFoundError("No CSV file found in dataset")

        file_path = csv_files[0]
        logger.info(f"Loading file: {file_path.name}")

        # load csv
        df = pd.read_csv(file_path)

        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df

    except Exception as e:
        logger.exception(f"Error while loading data: {e}")
        raise
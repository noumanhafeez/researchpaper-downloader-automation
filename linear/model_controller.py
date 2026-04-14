from linear_regression.src.pipeline import run_pipeline
from linear_regression.src.prediction import predict
from linear_regression.utils.logger import get_logger

logger = get_logger("main", "logs/main.log")


def runner(mode="train", samples=None):
    """
    Main entry point for Linear Regression project

    Args:
        mode (str): "train" or "predict"
        samples (list): input values for prediction
    """

    try:
        logger.info(f"Running mode: {mode}")

        if mode == "train":
            run_pipeline()
            logger.info("Training completed successfully")

        elif mode == "predict":
            if samples is None:
                raise ValueError("samples must be provided for prediction mode")

            preds = predict(samples)
            logger.info(f"Predictions: {preds}")
            print("Predictions:", preds)

        else:
            raise ValueError("Invalid mode. Use 'train' or 'predict'")

    except Exception as e:
        logger.exception(f"Error in main: {e}")
        raise

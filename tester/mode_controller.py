from decision_tree.src.train_app import train_app
from decision_tree.src.predict_app import predict_app
from decision_tree.utils.logger import get_logger

logger = get_logger("main", "logs/main.log")


def main(mode="predict", samples=None):
    """
    Main controller for application
    """
    try:
        logger.info("===== Application Started =====")
        logger.info(f"Mode: {mode}")

        if mode == "train":
            train_app()

        elif mode == "predict":
            if samples is None:
                raise ValueError("Samples are required for prediction mode")

            predict_app(samples)

        else:
            raise ValueError("Invalid mode. Use 'train' or 'predict'")

        logger.info("===== Application Finished =====")

    except Exception as e:
        logger.exception(f"Application failed: {e}")
        raise
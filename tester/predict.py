from decision_tree.src.prediction import predict_sample
from decision_tree.src.model_save import load_model
from decision_tree.utils.logger import get_logger

logger = get_logger("main", "logs/main.log")


def predict_app(samples):
    """
    Load saved model and only run prediction
    """
    try:
        logger.info("===== PREDICTION MODE STARTED =====")

        tree = load_model()
        logger.info("Model loaded successfully")

        predictions = []

        for i, sample in enumerate(samples):

            pred = predict_sample(tree, sample)

            if int(pred) == 0:
                result = "edible"
            elif int(pred) == 1:
                result = "poisonous"
            else:
                result = "unknown"

            print(f"Sample {i+1}: {result}")

            predictions.append(result)

        logger.info("===== PREDICTION COMPLETED =====")

        return predictions

    except Exception as e:
        logger.exception(f"Prediction failed: {e}")
        raise
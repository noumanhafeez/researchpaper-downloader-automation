from ..utils.logger import get_logger
from decision_tree.src.information_gain import compute_information_gain

logger = get_logger("feature_selection", "logs/feature_selection.log")


def get_best_feature(df, target):
    """
    Select feature with highest information gain
    """
    try:
        logger.info("Selecting best feature")

        features = [col for col in df.columns if col != target]

        ig_values = {}
        for feature in features:
            ig_values[feature] = compute_information_gain(df, feature, target)

        best_feature = max(ig_values, key=ig_values.get)

        logger.info(f"Best feature: {best_feature}")
        return best_feature

    except Exception as e:
        logger.exception(f"Error selecting best feature: {e}")
        raise
from ..utils.logger import get_logger
from decision_tree.src.entropy import compute_entropy
import numpy as np

logger = get_logger("information_gain", "logs/information_gain.log")


def compute_information_gain(df, feature, target):
    """
    Compute information gain for a feature
    """
    try:
        logger.info(f"Computing IG for feature: {feature}")

        total_entropy = compute_entropy(df[target])

        values, counts = np.unique(df[feature], return_counts=True)

        weighted_entropy = 0

        for v, count in zip(values, counts):
            subset = df[df[feature] == v]
            subset_entropy = compute_entropy(subset[target])

            weight = count / len(df)
            weighted_entropy += weight * subset_entropy

        ig = total_entropy - weighted_entropy

        logger.info(f"IG({feature}) = {ig:.4f}")
        return ig

    except Exception as e:
        logger.exception(f"Error computing IG: {e}")
        raise
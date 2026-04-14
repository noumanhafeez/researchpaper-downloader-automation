import numpy as np
from ..utils.logger import get_logger

logger = get_logger("entropy", "logs/entropy.log")


def compute_entropy(y):
    """
    Compute entropy of target column
    """
    try:
        logger.info("Computing entropy")

        values, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()

        entropy = -np.sum(probabilities * np.log2(probabilities))

        logger.info(f"Entropy: {entropy:.4f}")
        return entropy

    except Exception as e:
        logger.exception(f"Error computing entropy: {e}")
        raise
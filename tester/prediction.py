from decision_tree.utils.logger import get_logger

logger = get_logger("prediction", "logs/prediction.log")


def predict_sample(tree: dict, sample: dict):
    """
    Predict class for a single sample
    """
    # loop until leaf node
    while isinstance(tree, dict):
        feature = next(iter(tree))  # get first key
        value = sample.get(feature)

        tree = tree[feature].get(value)

        if tree is None:
            return None  # unseen value

    return tree


def predict(tree: dict, samples):
    """
    Predict for multiple samples

    Args:
        tree (dict)
        samples (list of dicts)

    Returns:
        list of predictions
    """
    return [predict_sample(tree, sample) for sample in samples]
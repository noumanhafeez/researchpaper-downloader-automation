from ..utils.logger import get_logger
import numpy as np
from decision_tree.src.feature_selection import get_best_feature

logger = get_logger("id3", "logs/id3.log")



def build_tree(df, target):
    """
    Recursively build decision tree using ID3
    """
    try:
        logger.info("Building tree")

        # ---- Stopping Condition 1 ----
        if len(np.unique(df[target])) == 1:
            label = df[target].iloc[0]
            logger.info(f"Leaf node: {label}")
            return label

        # ---- Stopping Condition 2 ----
        if len(df.columns) == 1:
            majority = df[target].mode()[0]
            logger.info(f"Majority leaf: {majority}")
            return majority

        # ---- Choose best feature ----
        best_feature = get_best_feature(df, target)

        tree = {best_feature: {}}

        # ---- Split dataset ----
        for value in np.unique(df[best_feature]):
            logger.info(f"Splitting {best_feature} = {value}")

            subset = df[df[best_feature] == value]

            # drop used feature
            subset = subset.drop(columns=[best_feature])

            subtree = build_tree(subset, target)

            tree[best_feature][value] = subtree

        return tree

    except Exception as e:
        logger.exception(f"Error building tree: {e}")
        raise
from decision_tree.src.pipeline import pipeline
from decision_tree.utils.logger import get_logger
from decision_tree.src.visualize_tree import tree_to_json
import json
from pathlib import Path

logger = get_logger("main", "logs/main.log")


output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

def train_app():
    try:
        logger.info("===== TRAINING MODE STARTED =====")

        tree, acc, precision, recall, f1 = pipeline()

        logger.info(f"Accuracy: {acc:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")

        logger.info("Model trained & saved successfully")

        tree_json = tree_to_json(tree)

        with open("outputs/decision_tree.json", "w") as f:
            json.dump(tree_json, f, indent=4)

        logger.info("Tree saved as JSON")

        logger.info("===== TRAINING COMPLETED =====")

    except Exception as e:
        logger.exception(f"Training failed: {e}")
        raise
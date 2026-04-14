from decision_tree.src.data_preprocessing import preprocess
from decision_tree.src.data_split import split_data
from decision_tree.src.recursive_id3 import build_tree
from decision_tree.src.prediction import predict
from decision_tree.utils.logger import get_logger
from decision_tree.src.model_save import save_model

logger = get_logger("pipeline", "logs/pipeline.log")


def accuracy(y_true, y_pred):
    return sum(yt == yp for yt, yp in zip(y_true, y_pred)) / len(y_true)

def precision_recall_f1(y_true, y_pred, positive_class=1):
    tp = sum((yt == positive_class and yp == positive_class) for yt, yp in zip(y_true, y_pred))
    fp = sum((yt != positive_class and yp == positive_class) for yt, yp in zip(y_true, y_pred))
    fn = sum((yt == positive_class and yp != positive_class) for yt, yp in zip(y_true, y_pred))

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

    return precision, recall, f1


def train_model(df):
    """
    Train ID3 model and return tree
    """
    X_train, X_test, y_train, y_test = split_data(df)

    train_df = X_train.copy()
    train_df["class"] = y_train

    tree = build_tree(train_df, target="class")

    return tree, X_test, y_test


def evaluate_model(tree, X_test, y_test):
    """
    Evaluate model using prediction module
    """
    X_test = X_test.to_dict(orient="records")
    y_pred = predict(tree, X_test)

    y_true = y_test.tolist()

    acc = accuracy(y_true, y_pred)
    precision, recall, f1 = precision_recall_f1(y_true, y_pred)

    return acc, precision, recall, f1, y_pred

def pipeline():
    try:
        logger.info("===== Starting ID3 Pipeline =====")

        df = preprocess()
        logger.info(f"Data shape: {df.shape}")

        tree, X_test, y_test = train_model(df)
        logger.info("Model trained successfully")

        acc, precision, recall, f1, y_pred = evaluate_model(tree, X_test, y_test)

        logger.info(f"Accuracy: {acc:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")

        save_model(tree)

        logger.info("===== Pipeline Completed =====")

        return tree, acc, precision, recall, f1

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        raise
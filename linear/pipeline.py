import os
import pickle

from linear_regression.src.model import SimpleLinearRegression
from linear_regression.src.sklearn_model import SklearnLinearRegressionModel
from linear_regression.src.data_split import split_data
from linear_regression.src.data_preprocessing import preprocess
from linear_regression.src.model_evaluation import evaluate_model
from linear_regression.utils.logger import get_logger

logger = get_logger("pipeline", "logs/pipeline.log")


def run_pipeline():
    try:
        logger.info("===== Starting Pipeline (Custom vs Sklearn) =====")

        df = preprocess()
        X_train, X_test, y_train, y_test = split_data(df)

        custom_model = SimpleLinearRegression()
        custom_model.fit(X_train, y_train)

        y_pred_custom = custom_model.predict(X_test)
        metrics_custom = evaluate_model(y_test, y_pred_custom)

        sklearn_model = SklearnLinearRegressionModel()
        sklearn_model.fit(X_train, y_train)

        y_pred_sklearn = sklearn_model.predict(X_test)
        metrics_sklearn = evaluate_model(y_test, y_pred_sklearn)

        logger.info(f"CUSTOM MODEL METRICS: {metrics_custom}")
        logger.info(f"SKLEARN MODEL METRICS: {metrics_sklearn}")

        print("\n===== MODEL COMPARISON =====")
        print("Custom Model:", metrics_custom)
        print("Sklearn Model:", metrics_sklearn)

        os.makedirs("artifacts", exist_ok=True)

        with open("artifacts/model.pkl", "wb") as f:
            pickle.dump(custom_model, f)

        logger.info("Pipeline completed successfully")

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        raise
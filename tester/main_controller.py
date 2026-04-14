from decision_tree.src.mode_controller import main as decision_tree_runner
from linear_regression.src.mode_controller import runner as linear_regression_runner
from linear_regression.src.visualize import plot_regression
from linear_regression.utils.logger import get_logger
from linear_regression.src.bonus_part.multifeature_pipeline import run_multi_pipeline

logger = get_logger("main", "logs/main.log")


def show_menu():
    print("\n==============================")
    print("        ML SYSTEM MENU        ")
    print("==============================")
    print("1. Decision Tree")
    print("2. Linear Regression")
    print("3. Visualize Linear Regression")
    print("4. Multi Feature Linear Regression")
    print("5. Exit")
    print("==============================")


def get_mode():
    print("\nChoose Mode:")
    print("1. Train")
    print("2. Predict")
    return input("Enter choice (1/2): ").strip()


def main(decision_tree_samples=None, linear_regression_samples=None):
    """
    Central controller for ML system
    """

    try:
        while True:

            show_menu()
            choice = input("Enter your choice: ").strip()

            if choice == "5":
                print("Exiting ML System... Goodbye 👋")
                break

            elif choice == "1":
                mode = "train" if get_mode() == "1" else "predict"

                samples = None
                if mode == "predict":
                    samples = decision_tree_samples

                decision_tree_runner(mode=mode, samples=samples)

            elif choice == "2":
                mode = "train" if get_mode() == "1" else "predict"

                samples = None
                if mode == "predict":
                    samples = linear_regression_samples

                linear_regression_runner(mode=mode, samples=samples)

            elif choice == "3":
                print("Launching regression visualization...")
                plot_regression()

            elif choice == "4":
                print("Running Multi Feature Linear Regression...")
                run_multi_pipeline()

            else:
                print("Invalid choice. Try again.")

    except Exception as e:
        logger.exception(f"Error in main controller: {e}")
        raise
from main_controller import main

if __name__ == "__main__":

    # You can define reusable sample sets here

    decision_tree_samples = [
        {"odor": 6, "gill-size": 1, "cap-surface": 2},
        {"odor": 3, "gill-size": 0, "cap-surface": 2},
        {"odor": 5, "gill-size": 0, "cap-surface": 2},
    ]

    linear_regression_samples = [50, 100, 150]

    main(
        decision_tree_samples=decision_tree_samples,
        linear_regression_samples=linear_regression_samples
    )
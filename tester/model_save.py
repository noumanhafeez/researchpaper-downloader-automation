import pickle


def save_model(tree, path="decision_tree/model/tree.pkl"):
    try:
        with open(path, "wb") as f:
            pickle.dump(tree, f)
        print(f"Model saved at {path}")
    except Exception as e:
        print(f"Error saving model: {e}")
        raise


def load_model(path="decision_tree/model/tree.pkl"):
    try:
        with open(path, "rb") as f:
            tree = pickle.load(f)
        print(f"Model loaded from {path}")
        return tree
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
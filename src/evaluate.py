import json

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from src import config


# scores a fitted model the same way for everyone and saves results/<name>.json
def evaluate(model, X, y, name, results_dir=None):
    results_dir = results_dir or config.RESULTS_DIR
    pred = model.predict(X)
    labels = config.CLASSES  # always all 4 classes, even if one is missing from y

    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=labels, zero_division=0)
    metrics = {
        "name": name,
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "per_class": {
            str(c): {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, c in enumerate(labels)
        },
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"{name}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_recall_fscore_support,
)

from src import config
from src.models import build_preprocessor


BATCH_SIZE = 1024

MODEL_DIR = (
    config.RESULTS_DIR
    / "neural_networks"
    / "deep"
)

MODEL_PATH = (
    MODEL_DIR
    / "deep_nn.keras"
)

PREPROCESSOR_PATH = (
    MODEL_DIR
    / "preprocessor.joblib"
)


def evaluate_model(model, X, y):
    probabilities = model.predict(
        X,
        batch_size=BATCH_SIZE,
        verbose=0,
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y,
            predictions,
            labels=[0, 1, 2, 3],
            zero_division=0,
        )
    )

    metrics = {
        "accuracy": float(
            accuracy_score(
                y,
                predictions,
            )
        ),
        "macro_f1": float(
            f1_score(
                y,
                predictions,
                labels=[0, 1, 2, 3],
                average="macro",
                zero_division=0,
            )
        ),
        "per_class": {
            str(i + 1): {
                "precision": float(
                    precision[i]
                ),
                "recall": float(
                    recall[i]
                ),
                "f1": float(
                    f1[i]
                ),
                "support": int(
                    support[i]
                ),
            }
            for i in range(4)
        },
        "confusion_matrix": confusion_matrix(
            y,
            predictions,
            labels=[0, 1, 2, 3],
        ).tolist(),
    }

    return metrics, predictions


def save_confusion_matrix(
    y,
    predictions,
):
    cm = confusion_matrix(
        y,
        predictions,
        labels=[0, 1, 2, 3],
        normalize="true",
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Severity 1",
            "Severity 2",
            "Severity 3",
            "Severity 4",
        ],
    )

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    display.plot(
        ax=ax,
        values_format=".2f",
    )

    ax.set_title(
        "Deep NN Test Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        MODEL_DIR
        / "test_confusion_matrix.png",
        dpi=200,
    )

    plt.close()


def main():
    print("Loading training sample and test data...")

    train = pd.read_parquet(
        config.TRAIN_SAMPLE_PARQUET
    )

    test = pd.read_parquet(
        config.TEST_PARQUET
    )

    features = config.MODEL_FEATURES

    print(
        f"Training sample rows: {len(train):,}"
    )

    print(
        f"Test rows: {len(test):,}"
    )

    # ---------------------------------------
    # Rebuild preprocessing using training
    # data only
    # ---------------------------------------

    print("\nFitting preprocessing pipeline...")

    preprocessor = build_preprocessor(
        features
    )

    preprocessor.fit(
        train[features]
    )

    X_test = preprocessor.transform(
        test[features]
    ).astype("float32")

    y_test = (
        test[config.TARGET]
        .to_numpy()
        .astype("int32")
        - 1
    )

    print(
        "Test shape:",
        X_test.shape,
    )

    # Save fitted preprocessing pipeline
    joblib.dump(
        preprocessor,
        PREPROCESSOR_PATH,
    )

    print(
        "Preprocessor saved to:",
        PREPROCESSOR_PATH,
    )

    # ---------------------------------------
    # Load selected Deep NN
    # ---------------------------------------

    print("\nLoading Deep NN...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    # ---------------------------------------
    # Final test evaluation
    # ---------------------------------------

    print("\nEvaluating untouched test data...")

    metrics, predictions = evaluate_model(
        model,
        X_test,
        y_test,
    )

    with open(
        MODEL_DIR / "test_metrics.json",
        "w",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    save_confusion_matrix(
        y_test,
        predictions,
    )

    # ---------------------------------------
    # Results
    # ---------------------------------------

    print("\n==============================")
    print("Deep NN Final Test Results")
    print("==============================")

    print(
        "Test Accuracy:",
        round(
            metrics["accuracy"],
            4,
        ),
    )

    print(
        "Test Macro-F1:",
        round(
            metrics["macro_f1"],
            4,
        ),
    )

    print("\nPer-Class Test F1:")

    for severity, values in (
        metrics["per_class"].items()
    ):
        print(
            f"Severity {severity}: "
            f"{values['f1']:.4f}"
        )

    print(
        "\nResults saved to:",
        MODEL_DIR,
    )


if __name__ == "__main__":
    main()
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src import config
from src.models import build_preprocessor
from src.neural_networks import (
    build_shallow_nn,
    build_wide_nn,
    build_deep_nn,
    build_regularized_deep_nn,
    build_regularized_deep_light_nn,
)


EPOCHS = 30
BATCH_SIZE = 1024
PATIENCE = 5


def get_model_builder(name):
    builders = {
        "shallow": build_shallow_nn,
        "wide": build_wide_nn,
        "deep": build_deep_nn,
        "regularized_deep": build_regularized_deep_nn,
        "regularized_deep_light": build_regularized_deep_light_nn,
    }

    return builders[name]


def evaluate_nn(model, X, y):
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

    return {
        "accuracy": float(
            accuracy_score(y, predictions)
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
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(4)
        },
        "confusion_matrix": confusion_matrix(
            y,
            predictions,
            labels=[0, 1, 2, 3],
        ).tolist(),
    }


def save_learning_curves(
    history,
    output_dir,
    model_name,
):
    history_df = pd.DataFrame(
        history.history
    )

    history_df.to_csv(
        output_dir / "history.csv",
        index=False,
    )

    display_name = (
        model_name
        .replace("_", " ")
        .title()
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        history_df["accuracy"],
        label="Training Accuracy",
    )

    plt.plot(
        history_df["val_accuracy"],
        label="Validation Accuracy",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{display_name} Accuracy")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / "accuracy_curve.png",
        dpi=200,
    )

    plt.close()

    plt.figure(figsize=(8, 5))

    plt.plot(
        history_df["loss"],
        label="Training Loss",
    )

    plt.plot(
        history_df["val_loss"],
        label="Validation Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{display_name} Loss")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / "loss_curve.png",
        dpi=200,
    )

    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=[
            "shallow",
            "wide",
            "deep",
            "regularized_deep",
            "regularized_deep_light",
        ],
        required=True,
    )

    args = parser.parse_args()

    model_name = args.model

    output_dir = (
        config.RESULTS_DIR
        / "neural_networks"
        / model_name
    )

    model_builder = get_model_builder(
        model_name
    )

    tf.keras.utils.set_random_seed(
        config.RANDOM_SEED
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading data...")

    train = pd.read_parquet(
        config.TRAIN_SAMPLE_PARQUET
    )

    val = pd.read_parquet(
        config.VAL_PARQUET
    )

    features = config.MODEL_FEATURES

    print(
        f"Training rows: {len(train):,}"
    )

    print(
        f"Validation rows: {len(val):,}"
    )

    print("\nPreprocessing data...")

    preprocessor = build_preprocessor(
        features
    )

    X_train = preprocessor.fit_transform(
        train[features]
    ).astype("float32")

    X_val = preprocessor.transform(
        val[features]
    ).astype("float32")

    y_train = (
        train[config.TARGET]
        .to_numpy()
        .astype("int32")
        - 1
    )

    y_val = (
        val[config.TARGET]
        .to_numpy()
        .astype("int32")
        - 1
    )

    print(
        "Training shape:",
        X_train.shape,
    )

    print(
        "Validation shape:",
        X_val.shape,
    )

    display_name = (
        model_name
        .replace("_", " ")
        .title()
    )

    print(
        f"\nBuilding {display_name}..."
    )

    model = model_builder(
        X_train.shape[1]
    )

    with open(
        output_dir / "model_summary.txt",
        "w",
    ) as f:
        model.summary(
            print_fn=lambda line: f.write(
                line + "\n"
            )
        )

    model.summary()

    early_stopping = (
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        )
    )

    checkpoint = (
        tf.keras.callbacks.ModelCheckpoint(
            output_dir
            / f"{model_name}_nn.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        )
    )

    print(
        f"\nTraining {display_name}..."
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(
            X_val,
            y_val,
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            early_stopping,
            checkpoint,
        ],
        verbose=1,
    )

    save_learning_curves(
        history,
        output_dir,
        model_name,
    )

    print(
        "\nEvaluating validation data..."
    )

    metrics = evaluate_nn(
        model,
        X_val,
        y_val,
    )

    with open(
        output_dir / "val_metrics.json",
        "w",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    print("\n==============================")
    print(f"{display_name} Results")
    print("==============================")

    print(
        "Validation Accuracy:",
        round(metrics["accuracy"], 4),
    )

    print(
        "Validation Macro-F1:",
        round(metrics["macro_f1"], 4),
    )

    print("\nPer-Class F1:")

    for severity, values in (
        metrics["per_class"].items()
    ):
        print(
            f"Severity {severity}: "
            f"{values['f1']:.4f}"
        )

    print(
        "\nResults saved to:",
        output_dir,
    )


if __name__ == "__main__":
    main()
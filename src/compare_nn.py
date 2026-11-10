import json

import matplotlib.pyplot as plt
import pandas as pd

from src import config


RESULTS_DIR = (
    config.RESULTS_DIR
    / "neural_networks"
)

MODELS = {
    "Shallow NN": "shallow",
    "Wide NN": "wide",
    "Deep NN": "deep",
    "Regularized Deep": "regularized_deep",
    "Regularized Deep Light": "regularized_deep_light",
}


def load_results():
    rows = []

    for display_name, folder_name in MODELS.items():
        path = (
            RESULTS_DIR
            / folder_name
            / "val_metrics.json"
        )

        with open(path, "r") as f:
            metrics = json.load(f)

        rows.append({
            "Model": display_name,
            "Validation Accuracy": metrics["accuracy"],
            "Validation Macro-F1": metrics["macro_f1"],
            "Severity 1 F1": metrics["per_class"]["1"]["f1"],
            "Severity 2 F1": metrics["per_class"]["2"]["f1"],
            "Severity 3 F1": metrics["per_class"]["3"]["f1"],
            "Severity 4 F1": metrics["per_class"]["4"]["f1"],
        })

    return pd.DataFrame(rows)


def save_summary_table(df):
    output_path = (
        RESULTS_DIR
        / "nn_comparison.csv"
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print("\nNeural Network Comparison")
    print("=" * 90)

    print(
        df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(
        "\nComparison table saved to:",
        output_path,
    )


def save_main_comparison_graph(df):
    plot_df = (
        df[
            [
                "Model",
                "Validation Accuracy",
                "Validation Macro-F1",
            ]
        ]
        .set_index("Model")
    )

    ax = plot_df.plot(
        kind="bar",
        figsize=(10, 6),
    )

    ax.set_title(
        "Neural Network Validation Performance"
    )

    ax.set_xlabel(
        "Model"
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_ylim(
        0,
        1,
    )

    plt.xticks(
        rotation=20,
        ha="right",
    )

    plt.legend(
        title="Metric"
    )

    plt.tight_layout()

    output_path = (
        RESULTS_DIR
        / "nn_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    print(
        "Main comparison graph saved to:",
        output_path,
    )


def save_macro_f1_graph(df):
    ordered = df.sort_values(
        "Validation Macro-F1",
        ascending=False,
    )

    plt.figure(
        figsize=(9, 5)
    )

    plt.bar(
        ordered["Model"],
        ordered["Validation Macro-F1"],
    )

    plt.title(
        "Neural Network Validation Macro-F1"
    )

    plt.xlabel(
        "Model"
    )

    plt.ylabel(
        "Macro-F1"
    )

    plt.ylim(
        0,
        max(
            ordered["Validation Macro-F1"]
        )
        + 0.05,
    )

    plt.xticks(
        rotation=20,
        ha="right",
    )

    for i, value in enumerate(
        ordered["Validation Macro-F1"]
    ):
        plt.text(
            i,
            value + 0.003,
            f"{value:.4f}",
            ha="center",
        )

    plt.tight_layout()

    output_path = (
        RESULTS_DIR
        / "nn_macro_f1.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    print(
        "Macro-F1 graph saved to:",
        output_path,
    )


def save_per_class_graph(df):
    columns = [
        "Severity 1 F1",
        "Severity 2 F1",
        "Severity 3 F1",
        "Severity 4 F1",
    ]

    plot_df = (
        df[
            ["Model"] + columns
        ]
        .set_index("Model")
    )

    ax = plot_df.plot(
        kind="bar",
        figsize=(11, 6),
    )

    ax.set_title(
        "Neural Network F1 by Severity"
    )

    ax.set_xlabel(
        "Model"
    )

    ax.set_ylabel(
        "F1 Score"
    )

    ax.set_ylim(
        0,
        1,
    )

    plt.xticks(
        rotation=20,
        ha="right",
    )

    plt.legend(
        title="Severity"
    )

    plt.tight_layout()

    output_path = (
        RESULTS_DIR
        / "nn_per_class_f1.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    print(
        "Per-class graph saved to:",
        output_path,
    )


def main():
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_results()

    save_summary_table(df)

    save_main_comparison_graph(df)

    save_macro_f1_graph(df)

    save_per_class_graph(df)

    best_model = df.loc[
        df["Validation Macro-F1"].idxmax()
    ]

    print("\nBest Neural Network")
    print("=" * 40)

    print(
        "Model:",
        best_model["Model"],
    )

    print(
        "Validation Accuracy:",
        f"{best_model['Validation Accuracy']:.4f}",
    )

    print(
        "Validation Macro-F1:",
        f"{best_model['Validation Macro-F1']:.4f}",
    )


if __name__ == "__main__":
    main()
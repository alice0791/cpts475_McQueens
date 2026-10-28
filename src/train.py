import argparse
import time

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ParameterGrid

from src import config
from src.evaluate import evaluate
from src.models import fit_model, make_model

QUICK_ROWS = 20_000


# try every setting in the grid, score on val, best macro-F1 first
def tune(kind, train, val, features, grid):
    rows = []
    for params in ParameterGrid(grid):
        start = time.time()
        pipe = fit_model(make_model(kind, features, **params), train[features], train[config.TARGET],
                         balanced=params.get("balanced", False))
        seconds = time.time() - start
        pred = pipe.predict(val[features])
        rows.append({
            **params,
            "val_macro_f1": f1_score(val[config.TARGET], pred, labels=config.CLASSES, average="macro", zero_division=0),
            "val_accuracy": accuracy_score(val[config.TARGET], pred),
            "fit_seconds": round(seconds, 1),
            "params": params,  # kept as a dict so None stays None
        })
        print(f"  {params}  macro-F1 {rows[-1]['val_macro_f1']:.4f}  ({seconds:.0f}s)")
    table = pd.DataFrame(rows)
    return table.sort_values("val_macro_f1", ascending=False, kind="stable").reset_index(drop=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="tune a model on val, then score the best one on val and test")
    parser.add_argument("--model", required=True, choices=list(config.PARAM_GRIDS))
    parser.add_argument("--extra", nargs="*", default=[], help="extra feature columns, like year Source")
    parser.add_argument("--quick", action="store_true", help=f"only use {QUICK_ROWS:,} rows, save to results/quick/")
    args = parser.parse_args(argv)

    features = config.MODEL_FEATURES + args.extra
    tag = args.model
    if args.extra:
        tag += "_plus_" + "_".join(e.lower() for e in args.extra)

    # paths read here so they follow config at run time
    data = config.PROCESSED_DIR
    sample = pd.read_parquet(data / config.TRAIN_SAMPLE_PARQUET.name)
    val = pd.read_parquet(data / config.VAL_PARQUET.name)
    test = pd.read_parquet(data / config.TEST_PARQUET.name)
    if args.quick:
        sample, val, test = sample.head(QUICK_ROWS), val.head(QUICK_ROWS), test.head(QUICK_ROWS)
        train = sample
        out_dir = config.RESULTS_DIR / "quick"
    else:
        train = pd.read_parquet(data / config.TRAIN_PARQUET.name)
        out_dir = config.RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"tuning {tag} on {len(sample):,} rows:")
    table = tune(args.model, sample, val, features, config.PARAM_GRIDS[args.model])
    table.drop(columns="params").to_csv(out_dir / f"tuning_{tag}.csv", index=False)

    best = table.loc[0, "params"]
    print(f"best: {best}, refitting on {len(train):,} rows")
    pipe = fit_model(make_model(args.model, features, **best), train[features], train[config.TARGET],
                     balanced=best.get("balanced", False))
    for name, part in [("val", val), ("test", test)]:
        m = evaluate(pipe, part[features], part[config.TARGET], f"{tag}_{name}", results_dir=out_dir)
        print(f"{name}: macro-F1 {m['macro_f1']:.4f}  accuracy {m['accuracy']:.4f}")


if __name__ == "__main__":
    main()

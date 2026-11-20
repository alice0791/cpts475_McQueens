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
    parser.add_argument("--feature-set", choices=list(config.FEATURE_SETS), default="combined",
                        help="experiment with one feature group or the combined feature set")
    parser.add_argument("--quick", action="store_true", help="use a subset of each split and save to results/quick/")
    parser.add_argument("--quick-rows", type=int, default=QUICK_ROWS,
                        help="rows per split for --quick (default: %(default)s)")
    parser.add_argument("--refit-sample", action="store_true",
                        help="refit the best model on the 1M training sample and score full val/test")
    args = parser.parse_args(argv)
    if args.quick_rows < 1:
        parser.error("--quick-rows must be a positive integer")
    if args.quick and args.refit_sample:
        parser.error("--quick cannot be combined with --refit-sample")

    features = config.FEATURE_SETS[args.feature_set] + args.extra
    tag = args.model
    if args.feature_set != "combined":
        tag += f"_only_{args.feature_set}"
    if args.extra:
        tag += "_plus_" + "_".join(e.lower() for e in args.extra)
    if args.refit_sample:
        tag += "_sampled"

    # paths read here so they follow config at run time
    data = config.PROCESSED_DIR
    sample = pd.read_parquet(data / config.TRAIN_SAMPLE_PARQUET.name)
    val = pd.read_parquet(data / config.VAL_PARQUET.name)
    test = pd.read_parquet(data / config.TEST_PARQUET.name)
    if args.quick:
        sample = sample.head(args.quick_rows)
        val = val.head(args.quick_rows)
        test = test.head(args.quick_rows)
        train = sample
        out_dir = config.RESULTS_DIR / "quick"
    elif args.refit_sample:
        train = sample
        out_dir = config.RESULTS_DIR
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

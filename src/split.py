import argparse

from sklearn.model_selection import train_test_split

from src import config
from src.data import clean, load_raw, load_raw_cached
from src.features import add_features


# 70/15/15 split, stratified so every split has the same severity mix
def make_splits(df, seed=config.RANDOM_SEED):
    train, rest = train_test_split(df, test_size=0.30, stratify=df[config.TARGET], random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest[config.TARGET], random_state=seed)
    return train, val, test


# smaller stratified piece of train for slow models
def sample_train(train, n=config.TRAIN_SAMPLE_SIZE, seed=config.RANDOM_SEED):
    if len(train) <= n:
        return train
    sample, _ = train_test_split(train, train_size=n, stratify=train[config.TARGET], random_state=seed)
    return sample


def save_splits(train, val, test, sample, out_dir=config.PROCESSED_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out_dir / config.TRAIN_PARQUET.name)
    val.to_parquet(out_dir / config.VAL_PARQUET.name)
    test.to_parquet(out_dir / config.TEST_PARQUET.name)
    sample.to_parquet(out_dir / config.TRAIN_SAMPLE_PARQUET.name)


def main(argv=None):
    parser = argparse.ArgumentParser(description="clean the raw data and save the train/val/test split")
    parser.add_argument("--nrows", type=int, help="only use the first n rows (for quick runs)")
    args = parser.parse_args(argv)

    # paths are read here, not as default args, so they follow config at run time
    if args.nrows:
        raw = load_raw(nrows=args.nrows, path=config.RAW_CSV)
    else:
        raw = load_raw_cached(path=config.RAW_CSV, cache=config.RAW_PARQUET)
    # quick runs go in their own folder so they never replace the real splits
    out_dir = config.PROCESSED_DIR / "quick" if args.nrows else config.PROCESSED_DIR
    stats = {}
    df = add_features(clean(raw, stats))
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / config.CLEAN_PARQUET.name)

    train, val, test = make_splits(df)
    sample = sample_train(train)
    save_splits(train, val, test, sample, out_dir=out_dir)

    print("cleaning:")
    for key, value in stats.items():
        print(f"  {key}: {value:,}")
    print("rows per split:")
    for name, part in [("train", train), ("val", val), ("test", test), ("train_sample", sample)]:
        counts = part[config.TARGET].value_counts().sort_index()
        print(f"  {name}: {len(part):,}  severity 1-4: {counts.tolist()}")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()

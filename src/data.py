import pandas as pd

from src import config


# read the raw CSV with the dtypes from config
def load_raw(nrows=None, path=config.RAW_CSV):
    return pd.read_csv(path, dtype=config.RAW_DTYPES, nrows=nrows)


# same as load_raw() but saves a parquet copy the first time, which loads much faster
def load_raw_cached(path=config.RAW_CSV, cache=config.RAW_PARQUET):
    if cache.exists():
        return pd.read_parquet(cache)
    df = load_raw(path=path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df

import warnings

import numpy as np
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


# cleans the raw data. no imputing or scaling here,
# that gets fit on train only. pass a dict as stats to get the row counts back.
def clean(df, stats=None):
    if stats is None:
        stats = {}
    df = df.copy()
    stats["rows_in"] = len(df)

    # some times have fractional seconds, so format="mixed"
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], format="mixed", errors="coerce")
    bad_time = df["Start_Time"].isna()
    stats["bad_start_time"] = int(bad_time.sum())
    df = df[~bad_time]

    no_sun = df["Sunrise_Sunset"].isna()
    stats["missing_sunrise_sunset"] = int(no_sun.sum())
    df = df[~no_sun]

    # ID makes every row unique, so ignore it when looking for copies
    exact = df.drop(columns="ID").duplicated()
    stats["exact_duplicates"] = int(exact.sum())
    df = df[~exact]

    same_key = df.duplicated(subset=config.DEDUP_KEY)
    stats["key_duplicates"] = int(same_key.sum())
    df = df[~same_key]

    drop = config.LEAKAGE_COLS + config.ID_COLS + config.OUT_OF_SCOPE_COLS
    if df["Turning_Loop"].nunique() <= 1:
        drop = drop + ["Turning_Loop"]
    else:
        warnings.warn("Turning_Loop is not constant, keeping it")
        df["Turning_Loop"] = df["Turning_Loop"].astype("int8")
    df = df.drop(columns=drop)

    df[config.ROAD_COLS] = df[config.ROAD_COLS].astype("int8")
    df["is_night"] = (df["Sunrise_Sunset"] == "Night").astype("int8")
    df = df.drop(columns="Sunrise_Sunset")

    # flag missing values before filling anything. precip missing usually means none
    for col, flag in config.MISSING_FLAGS.items():
        df[flag] = df[col].isna().astype("int8")
    df["Precipitation(in)"] = df["Precipitation(in)"].fillna(0)

    for col, (lo, hi) in config.CLIP_BOUNDS.items():
        df[col] = df[col].clip(lo, hi)

    df["weather_group"] = group_weather(df["Weather_Condition"])
    df["Weather_Condition"] = df["Weather_Condition"].astype("string").fillna("Unknown").astype("category")
    df["Wind_Direction"] = normalize_wind_direction(df["Wind_Direction"])

    stats["rows_out"] = len(df)
    return df


# map the ~140 weather labels to a few groups using config.WEATHER_GROUPS
def group_weather(s):
    text = s.astype("string").str.lower()
    conditions = [text.str.contains("|".join(words), regex=True).fillna(False)
                  for words in config.WEATHER_GROUPS.values()]
    groups = np.select(conditions, list(config.WEATHER_GROUPS), default="Other")
    groups = np.where(text.isna(), "Unknown", groups)
    return pd.Series(groups, index=s.index, dtype="category")


# CALM -> Calm, North -> N, etc. missing or anything odd -> Unknown
def normalize_wind_direction(s):
    fixed = s.astype("string").replace(config.WIND_FIXES)
    fixed = fixed.where(fixed.isin(config.WIND_DIRECTIONS), "Unknown")
    return fixed.fillna("Unknown").astype("category")

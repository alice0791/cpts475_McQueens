import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import clean, group_weather, load_raw, load_raw_cached, normalize_wind_direction


def test_load_raw_dtypes(raw_csv):
    df = load_raw(path=raw_csv)
    assert len(df) == 200
    assert df["Severity"].dtype == "int8"
    assert df["Amenity"].dtype == "boolean"
    assert df["Source"].dtype == "category"


def test_load_raw_nrows(raw_csv):
    assert len(load_raw(nrows=50, path=raw_csv)) == 50


def test_load_raw_cached_writes_then_reuses(raw_csv, tmp_path):
    cache = tmp_path / "raw.parquet"
    first = load_raw_cached(path=raw_csv, cache=cache)
    assert cache.exists()

    raw_csv.unlink()  # second call has to come from the cache
    second = load_raw_cached(path=raw_csv, cache=cache)
    pd.testing.assert_frame_equal(first, second)


def test_clean_drops_leakage_and_id_columns(raw_df):
    out = clean(raw_df)
    dropped = config.LEAKAGE_COLS + config.ID_COLS + config.OUT_OF_SCOPE_COLS
    assert not set(dropped) & set(out.columns)
    assert "Sunrise_Sunset" not in out.columns
    assert "Turning_Loop" not in out.columns


def test_clean_parses_fractional_seconds(raw_df):
    out = clean(raw_df)
    assert out.loc[0, "Start_Time"] == pd.Timestamp("2016-02-08 05:46:00")


def test_clean_drops_bad_rows_and_duplicates(raw_df):
    stats = {}
    out = clean(raw_df, stats)
    for i in (1, 2, 4, 5, 7):
        assert i not in out.index
    assert stats == {
        "rows_in": 200,
        "bad_start_time": 1,
        "missing_sunrise_sunset": 1,
        "exact_duplicates": 2,
        "key_duplicates": 1,
        "rows_out": 195,
    }
    assert len(out) == 195


def test_clean_road_and_night_are_int01(raw_df):
    out = clean(raw_df)
    for col in config.ROAD_COLS + ["is_night"]:
        assert out[col].dtype == "int8"
        assert set(out[col].unique()) <= {0, 1}


def test_clean_does_not_mutate_input(raw_df):
    before = raw_df.copy()
    clean(raw_df)
    pd.testing.assert_frame_equal(raw_df, before)


def test_turning_loop_kept_with_warning_if_not_constant(raw_df):
    raw_df.loc[0, "Turning_Loop"] = True
    with pytest.warns(UserWarning, match="Turning_Loop"):
        out = clean(raw_df)
    assert out["Turning_Loop"].dtype == "int8"


@pytest.mark.parametrize("raw, group", [
    ("Heavy T-Storm", "Thunderstorm"),
    ("Thunderstorms and Snow", "Thunderstorm"),
    ("Light Snow", "Snow/Ice"),
    ("Light Freezing Rain", "Snow/Ice"),
    ("Light Rain", "Rain"),
    ("Haze", "Fog/Haze"),
    ("Mostly Cloudy / Windy", "Windy/Dust"),
    ("Squalls", "Windy/Dust"),
    ("Partly Cloudy", "Cloudy"),
    ("Fair", "Clear"),
    ("Volcanic Ash", "Other"),
    (np.nan, "Unknown"),
])
def test_group_weather_buckets(raw, group):
    assert group_weather(pd.Series([raw], dtype="category"))[0] == group


def test_wind_direction_normalized():
    raw = pd.Series(["CALM", "Calm", "VAR", "Variable", "North", "West", "SSW", np.nan], dtype="category")
    out = normalize_wind_direction(raw)
    assert out.tolist() == ["Calm", "Calm", "Variable", "Variable", "N", "W", "SSW", "Unknown"]


def test_precip_filled_and_flagged(raw_df):
    out = clean(raw_df)
    was_missing = raw_df.loc[out.index, "Precipitation(in)"].isna()
    assert out["Precipitation(in)"].isna().sum() == 0
    assert (out["precip_missing"] == was_missing.astype("int8")).all()


def test_other_numeric_nan_kept(raw_df):
    out = clean(raw_df)
    assert out["Temperature(F)"].isna().sum() == 4
    assert (out["temp_missing"] == out["Temperature(F)"].isna().astype("int8")).all()


def test_flags_present_without_missing(raw_df):
    raw_df[config.NUMERIC_WEATHER] = raw_df[config.NUMERIC_WEATHER].fillna(1.0)
    out = clean(raw_df)
    for flag in config.MISSING_FLAGS.values():
        assert flag in out.columns
        assert out[flag].sum() == 0


def test_outliers_clipped(raw_df):
    out = clean(raw_df)
    for col, (lo, hi) in config.CLIP_BOUNDS.items():
        assert out[col].min() >= lo
        assert out[col].max() <= hi


def test_clean_categories_have_no_nan(raw_df):
    out = clean(raw_df)
    for col in ["weather_group", "Weather_Condition", "Wind_Direction"]:
        assert out[col].isna().sum() == 0

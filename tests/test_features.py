import pandas as pd

from src import config
from src.data import clean
from src.features import add_features, add_road_features, add_time_features


def small_frame():
    df = pd.DataFrame({"Start_Time": pd.to_datetime([
        "2023-03-04 08:30",  # saturday morning
        "2023-03-01 12:00",  # wednesday noon
        "2023-03-01 19:59",
        "2023-03-01 20:00",
    ])})
    for col in config.ROAD_COLS:
        df[col] = pd.Series([0, 0, 0, 0], dtype="int8")
    df["Crossing"] = pd.Series([1, 0, 1, 0], dtype="int8")
    df["Stop"] = pd.Series([1, 0, 0, 0], dtype="int8")
    df["Junction"] = pd.Series([1, 1, 0, 0], dtype="int8")
    return df


def test_time_features():
    out = add_time_features(small_frame())
    assert out["hour"].tolist() == [8, 12, 19, 20]
    assert out["day_of_week"].tolist() == [5, 2, 2, 2]
    assert out["month"].tolist() == [3, 3, 3, 3]
    assert out["year"].tolist() == [2023] * 4
    assert out["is_weekend"].tolist() == [1, 0, 0, 0]
    assert out["is_rush_hour"].tolist() == [1, 0, 1, 0]


def test_road_feature_count():
    out = add_road_features(small_frame())
    assert out["road_feature_count"].tolist() == [3, 1, 1, 0]


def test_features_do_not_mutate_input():
    df = small_frame()
    add_features(df)
    assert "hour" not in df.columns


def test_every_feature_group_column_exists(raw_df):
    out = add_features(clean(raw_df))
    for group, cols in config.FEATURE_GROUPS.items():
        missing = set(cols) - set(out.columns)
        assert not missing, f"{group} is missing {missing}"

from src import config

RUSH_HOURS = [7, 8, 9, 16, 17, 18, 19]


# hour, day of week (0 = monday), month, year, weekend and rush hour flags
def add_time_features(df):
    df = df.copy()
    t = df["Start_Time"].dt
    df["hour"] = t.hour.astype("int8")
    df["day_of_week"] = t.dayofweek.astype("int8")
    df["month"] = t.month.astype("int8")
    df["year"] = t.year.astype("int16")
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")
    df["is_rush_hour"] = df["hour"].isin(RUSH_HOURS).astype("int8")
    return df


# how many road features are near the accident
def add_road_features(df):
    df = df.copy()
    df["road_feature_count"] = df[config.ROAD_COLS].sum(axis=1).astype("int8")
    return df


def add_features(df):
    return add_road_features(add_time_features(df))

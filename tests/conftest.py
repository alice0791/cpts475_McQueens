import numpy as np
import pandas as pd
import pytest

from src import config

WEATHER = ["Fair", "Cloudy", "Light Rain", "Heavy T-Storm", "Light Snow", "Fog",
           "Haze", "Blowing Dust / Windy", "Mostly Cloudy / Windy", "Squalls", np.nan]
WIND = ["CALM", "Calm", "VAR", "Variable", "North", "N", "SSW", "West", np.nan]


# fake data shaped like the raw CSV, with some messy rows on purpose
def make_raw(n=200, seed=0):
    rng = np.random.default_rng(seed)

    # first 8 rows are severity 2 so the duplicate rows below don't change the counts
    rest = np.repeat([1, 2, 3, 4], [20, 132, 30, 10])
    severity = np.concatenate([np.full(8, 2), rng.permutation(rest)])

    # one row every ~350 hours covers 2016-2023 with no repeated times
    times = pd.date_range("2016-01-01", periods=n, freq="350h")

    df = pd.DataFrame({
        "ID": [f"A-{i}" for i in range(n)],
        "Source": rng.choice(["Source1", "Source2", "Source3"], n),
        "Severity": severity,
        "Start_Time": times.strftime("%Y-%m-%d %H:%M:%S"),
        "End_Time": (times + pd.Timedelta("1h")).strftime("%Y-%m-%d %H:%M:%S"),
        "Start_Lat": rng.uniform(25, 49, n),
        "Start_Lng": rng.uniform(-124, -67, n),
        "End_Lat": rng.uniform(25, 49, n),
        "End_Lng": rng.uniform(-124, -67, n),
        "Distance(mi)": rng.uniform(0, 5, n),
        "Description": [f"accident {i}" for i in range(n)],
        "Street": "Main St",
        "City": "Pullman",
        "County": "Whitman",
        "State": rng.choice(["WA", "CA", "TX"], n),
        "Zipcode": "99163",
        "Country": "US",
        "Timezone": "US/Pacific",
        "Airport_Code": "KPUW",
        "Weather_Timestamp": times.strftime("%Y-%m-%d %H:%M:%S"),
        "Temperature(F)": rng.uniform(10, 95, n),
        "Wind_Chill(F)": rng.uniform(0, 90, n),
        "Humidity(%)": rng.uniform(10, 100, n),
        "Pressure(in)": rng.uniform(29, 30.5, n),
        "Visibility(mi)": rng.uniform(0, 10, n),
        "Wind_Direction": rng.choice(np.array(WIND, dtype=object), n),
        "Wind_Speed(mph)": rng.uniform(0, 30, n),
        "Precipitation(in)": rng.uniform(0, 0.5, n),
        "Weather_Condition": rng.choice(np.array(WEATHER, dtype=object), n),
    })
    for col in config.ROAD_COLS:
        df[col] = rng.random(n) < 0.2
    df["Turning_Loop"] = False
    for col in ["Sunrise_Sunset", "Civil_Twilight", "Nautical_Twilight", "Astronomical_Twilight"]:
        df[col] = rng.choice(["Day", "Night"], n)

    # the messy rows
    df.loc[0, "Start_Time"] = "2016-02-08 05:46:00.000000000"
    df.loc[1, "Start_Time"] = "not a date"
    df.loc[2, "Sunrise_Sunset"] = np.nan
    df.loc[10, "Temperature(F)"] = 200
    df.loc[11, "Wind_Speed(mph)"] = 250
    df.loc[rng.random(n) < 0.2, "Precipitation(in)"] = np.nan
    df.loc[12:15, "Temperature(F)"] = np.nan
    df.loc[7, config.DEDUP_KEY] = df.loc[6, config.DEDUP_KEY]  # same time and place as row 6
    for i in (4, 5):  # exact copies of row 3 apart from ID (done last so nothing changes them)
        df.loc[i, df.columns != "ID"] = df.loc[3, df.columns != "ID"]

    return df[list(config.RAW_DTYPES)].astype(config.RAW_DTYPES)


@pytest.fixture
def raw_df():
    return make_raw()


@pytest.fixture
def raw_csv(tmp_path, raw_df):
    path = tmp_path / "raw.csv"
    raw_df.to_csv(path, index=False)
    return path

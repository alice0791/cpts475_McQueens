# settings shared by the whole project
import os
from pathlib import Path

# paths
ROOT = Path(__file__).resolve().parents[1]
# set CPTS475_DATA_DIR to use a different data folder
DATA_DIR = Path(os.environ.get("CPTS475_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_CSV = RAW_DIR / "US_Accidents_March23.csv"
RAW_PARQUET = PROCESSED_DIR / "raw.parquet"
CLEAN_PARQUET = PROCESSED_DIR / "clean.parquet"
TRAIN_PARQUET = PROCESSED_DIR / "train.parquet"
VAL_PARQUET = PROCESSED_DIR / "val.parquet"
TEST_PARQUET = PROCESSED_DIR / "test.parquet"
TRAIN_SAMPLE_PARQUET = PROCESSED_DIR / "train_sample_1m.parquet"

RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "reports" / "figures"


RANDOM_SEED = 42
TARGET = "Severity"
CLASSES = [1, 2, 3, 4]
TRAIN_SAMPLE_SIZE = 1_000_000  # smaller train set for KNN / neural nets

# road features (Turning_Loop is handled in clean())
ROAD_COLS = [
    "Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit",
    "Railway", "Roundabout", "Station", "Stop", "Traffic_Calming",
    "Traffic_Signal",
]

# only known after the accident, so they leak severity
LEAKAGE_COLS = ["End_Time", "End_Lat", "End_Lng", "Distance(mi)", "Description"]

ID_COLS = ["ID", "Country"]

# not used as features (lat/lng are dropped after removing duplicates)
OUT_OF_SCOPE_COLS = [
    "Street", "City", "County", "Zipcode", "Airport_Code", "Weather_Timestamp",
    "Timezone", "Wind_Chill(F)", "Civil_Twilight", "Nautical_Twilight",
    "Astronomical_Twilight", "Start_Lat", "Start_Lng",
]

# same time + same place = same accident
DEDUP_KEY = ["Start_Time", "Start_Lat", "Start_Lng"]

NUMERIC_WEATHER = [
    "Temperature(F)", "Humidity(%)", "Pressure(in)", "Visibility(mi)",
    "Wind_Speed(mph)", "Precipitation(in)",
]

# column -> name of its missing flag
MISSING_FLAGS = {
    "Precipitation(in)": "precip_missing",
    "Temperature(F)": "temp_missing",
    "Humidity(%)": "humidity_missing",
    "Pressure(in)": "pressure_missing",
    "Visibility(mi)": "visibility_missing",
    "Wind_Speed(mph)": "wind_speed_missing",
}

# values outside these ranges get clipped
CLIP_BOUNDS = {
    "Temperature(F)": (-40, 130),
    "Wind_Speed(mph)": (0, 100),
    "Visibility(mi)": (0, 10),
    "Humidity(%)": (0, 100),
    "Pressure(in)": (25, 32),
}

# weather label -> group, first match wins (so "Light Rain / Windy" is Rain)
WEATHER_GROUPS = {
    "Thunderstorm": ["thunder", "t-storm", "tstorm"],
    "Snow/Ice": ["snow", "sleet", "ice", "freezing", "wintry", "hail"],
    "Rain": ["rain", "drizzle", "shower"],
    "Fog/Haze": ["fog", "haze", "mist", "smoke"],
    "Windy/Dust": ["windy", "dust", "sand", "squall", "tornado", "funnel"],
    "Cloudy": ["cloud", "overcast"],
    "Clear": ["clear", "fair"],
}

# different spellings of the same wind direction
WIND_FIXES = {"CALM": "Calm", "VAR": "Variable", "North": "N", "South": "S", "East": "E", "West": "W"}
WIND_DIRECTIONS = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW", "Calm", "Variable",
]

# year is left out for now until we check it in EDA
FEATURE_GROUPS = {
    "weather": NUMERIC_WEATHER + list(MISSING_FLAGS.values())
    + ["weather_group", "Wind_Direction", "is_night"],
    "time": ["hour", "day_of_week", "month", "is_weekend", "is_rush_hour"],
    "road": ROAD_COLS + ["road_feature_count"],
}

# features for lr / nb / dt, no year or Source
MODEL_FEATURES = FEATURE_GROUPS["weather"] + FEATURE_GROUPS["time"] + FEATURE_GROUPS["road"]
# one-hot these, everything else is a number
CATEGORICAL_FEATURES = ["weather_group", "Wind_Direction", "Source"]

# settings to try for each model
PARAM_GRIDS = {
    "lr": {"model__C": [0.01, 0.1, 1.0], "model__class_weight": [None, "balanced"]},
    "nb": {"model__alpha": [0.01, 0.1, 1.0], "balanced": [False, True]},
    "dt": {
        "model__max_depth": [5, 10, 15, 20, None],
        "model__min_samples_leaf": [1, 50, 500],
        "model__class_weight": [None, "balanced"],
    },
}

# dtypes for reading the CSV (times are parsed later in clean())
_STRING = [
    "ID", "Start_Time", "End_Time", "Description", "Street", "City", "County",
    "Zipcode", "Airport_Code", "Weather_Timestamp",
]
_FLOAT = [
    "Start_Lat", "Start_Lng", "End_Lat", "End_Lng", "Distance(mi)",
    "Temperature(F)", "Wind_Chill(F)", "Humidity(%)", "Pressure(in)",
    "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)",
]
_CATEGORY = [
    "Source", "State", "Country", "Timezone", "Wind_Direction",
    "Weather_Condition", "Sunrise_Sunset", "Civil_Twilight",
    "Nautical_Twilight", "Astronomical_Twilight",
]
_BOOL = ROAD_COLS + ["Turning_Loop"]

RAW_DTYPES = {
    **{c: "string" for c in _STRING},
    **{c: "float32" for c in _FLOAT},
    **{c: "category" for c in _CATEGORY},
    **{c: "boolean" for c in _BOOL},
    "Severity": "int8",
}

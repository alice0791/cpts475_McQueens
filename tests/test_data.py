import pandas as pd

from src.data import load_raw, load_raw_cached


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

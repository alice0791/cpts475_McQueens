import pandas as pd
import pytest

from src import config
from src.data import clean
from src.features import add_features
from src.split import main, make_splits, sample_train, save_splits


@pytest.fixture
def featured(raw_df):
    # 10 copies of the cleaned fake data (~2000 rows) so every class has enough rows
    one = add_features(clean(raw_df))
    copies = [one.set_axis(one.index + i * 1000) for i in range(10)]
    return pd.concat(copies)


def class_share(df):
    return df[config.TARGET].value_counts(normalize=True).sort_index()


def test_split_proportions(featured):
    train, val, test = make_splits(featured)
    n = len(featured)
    assert abs(len(train) - 0.70 * n) <= 1
    assert abs(len(val) - 0.15 * n) <= 1
    assert abs(len(test) - 0.15 * n) <= 1


def test_split_stratified(featured):
    full = class_share(featured)
    for part in make_splits(featured):
        diff = (class_share(part) - full).abs()
        assert (diff < 0.02).all()


def test_no_row_overlap(featured):
    train, val, test = make_splits(featured)
    assert not set(train.index) & set(val.index)
    assert not set(train.index) & set(test.index)
    assert not set(val.index) & set(test.index)
    assert len(train) + len(val) + len(test) == len(featured)
    assert set(train.index) | set(val.index) | set(test.index) == set(featured.index)


def test_split_deterministic(featured):
    first = make_splits(featured)
    second = make_splits(featured)
    for a, b in zip(first, second):
        assert a.index.equals(b.index)


def test_sample_train_small_returns_all(featured):
    train, _, _ = make_splits(featured)
    assert len(sample_train(train, n=len(train) + 10)) == len(train)


def test_sample_train_stratified(featured):
    train, _, _ = make_splits(featured)
    sample = sample_train(train, n=500)
    assert len(sample) == 500
    assert ((class_share(sample) - class_share(train)).abs() < 0.02).all()


def test_save_splits_writes_parquet(featured, tmp_path):
    train, val, test = make_splits(featured)
    sample = sample_train(train, n=500)
    save_splits(train, val, test, sample, out_dir=tmp_path)
    for name, df in [("train", train), ("val", val), ("test", test), ("train_sample_1m", sample)]:
        back = pd.read_parquet(tmp_path / f"{name}.parquet")
        assert back.index.equals(df.index)


def test_nrows_run_does_not_touch_real_splits(raw_csv, tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    monkeypatch.setattr(config, "RAW_CSV", raw_csv)
    monkeypatch.setattr(config, "PROCESSED_DIR", processed)
    main(["--nrows", "200"])
    assert (processed / "quick" / "train.parquet").exists()
    assert (processed / "quick" / "clean.parquet").exists()
    assert not (processed / "train.parquet").exists()

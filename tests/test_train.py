import pandas as pd
import pytest

from src import config
from src.data import clean
from src.features import add_features
from src.split import make_splits, sample_train, save_splits
from src.train import main, tune


@pytest.fixture
def splits(raw_df):
    # 10 copies of the fake data so every class shows up in val
    one = add_features(clean(raw_df))
    featured = pd.concat([one.set_axis(one.index + i * 1000) for i in range(10)])
    return make_splits(featured)


# fake data/processed and results folders, and a 1-setting grid so it runs fast
@pytest.fixture
def fake_project(splits, tmp_path, monkeypatch):
    train, val, test = splits
    save_splits(train, val, test, sample_train(train, n=500), out_dir=tmp_path / "processed")
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setitem(config.PARAM_GRIDS, "dt", {"model__max_depth": [3]})
    return tmp_path / "results"


def test_tune_returns_sorted_table(splits):
    train, val, _ = splits
    table = tune("dt", train, val, config.MODEL_FEATURES, {"model__max_depth": [2, 4]})
    assert len(table) == 2
    assert table["val_macro_f1"].is_monotonic_decreasing
    assert {"model__max_depth", "val_accuracy", "fit_seconds"} <= set(table.columns)


def test_tune_is_repeatable(splits):
    train, val, _ = splits
    grid = {"model__max_depth": [2, 4]}
    first = tune("dt", train, val, config.MODEL_FEATURES, grid)
    second = tune("dt", train, val, config.MODEL_FEATURES, grid)
    assert first["val_macro_f1"].tolist() == second["val_macro_f1"].tolist()


def test_quick_run_writes_to_quick_folder(fake_project):
    main(["--model", "dt", "--quick"])
    quick = fake_project / "quick"
    assert (quick / "tuning_dt.csv").exists()
    assert (quick / "dt_val.json").exists()
    assert (quick / "dt_test.json").exists()
    # nothing outside quick/
    assert [p.name for p in fake_project.iterdir()] == ["quick"]


def test_extra_features_change_tag(fake_project):
    main(["--model", "dt", "--quick", "--extra", "year"])
    assert (fake_project / "quick" / "dt_plus_year_test.json").exists()

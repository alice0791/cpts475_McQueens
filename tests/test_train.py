import json
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


def test_quick_rows_limits_each_evaluation_split(fake_project):
    main(["--model", "dt", "--quick", "--quick-rows", "20"])
    with open(fake_project / "quick" / "dt_test.json") as f:
        result = json.load(f)
    assert sum(c["support"] for c in result["per_class"].values()) == 20


def test_refit_sample_uses_sample_and_scores_full_splits(fake_project, splits, monkeypatch):
    _, _, test = splits
    import src.train as train_module

    sample = pd.read_parquet(config.PROCESSED_DIR / config.TRAIN_SAMPLE_PARQUET.name)
    fit_sizes = []
    original_fit_model = train_module.fit_model

    def track_fit_size(pipe, X, y, balanced=False):
        fit_sizes.append(len(X))
        return original_fit_model(pipe, X, y, balanced=balanced)

    monkeypatch.setattr(train_module, "fit_model", track_fit_size)
    main(["--model", "dt", "--refit-sample"])

    assert fit_sizes and set(fit_sizes) == {len(sample)}
    with open(fake_project / "dt_sampled_test.json") as f:
        result = json.load(f)
    assert sum(c["support"] for c in result["per_class"].values()) == len(test)
    assert (fake_project / "dt_sampled_val.json").exists()


def test_extra_features_change_tag(fake_project):
    main(["--model", "dt", "--quick", "--extra", "year"])
    assert (fake_project / "quick" / "dt_plus_year_test.json").exists()


def test_feature_set_changes_tag(fake_project):
    main(["--model", "knn", "--quick", "--feature-set", "weather"])
    assert (fake_project / "quick" / "knn_only_weather_test.json").exists()

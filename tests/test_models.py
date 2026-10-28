import numpy as np
import pytest

from src import config
from src.data import clean
from src.features import add_features
from src.models import fit_model, make_model


@pytest.fixture
def featured(raw_df):
    return add_features(clean(raw_df))


def xy(df):
    return df[config.MODEL_FEATURES], df[config.TARGET]


@pytest.mark.parametrize("kind", ["lr", "nb", "dt"])
def test_each_kind_fits_and_predicts(featured, kind):
    X, y = xy(featured)
    pipe = fit_model(make_model(kind, config.MODEL_FEATURES), X, y)
    pred = pipe.predict(X)
    assert len(pred) == len(X)
    assert set(pred) <= set(config.CLASSES)


def test_unseen_category_in_predict(featured):
    X, y = xy(featured)
    pipe = fit_model(make_model("lr", config.MODEL_FEATURES), X, y)
    X_new = X.copy()
    X_new["Wind_Direction"] = X_new["Wind_Direction"].astype(object)
    X_new.iloc[:5, X_new.columns.get_loc("Wind_Direction")] = "XYZ"
    pipe.predict(X_new)  # should not raise


def test_all_nan_numeric_column_keeps_shape(featured):
    X, y = xy(featured)
    normal = fit_model(make_model("lr", config.MODEL_FEATURES), X, y)
    X_nan = X.copy()
    X_nan["Temperature(F)"] = np.nan
    empty = fit_model(make_model("lr", config.MODEL_FEATURES), X_nan, y)
    assert empty[:-1].transform(X_nan).shape[1] == normal[:-1].transform(X).shape[1]


def test_balanced_weights_reach_nb(featured):
    X, y = xy(featured)
    plain = fit_model(make_model("nb", config.MODEL_FEATURES), X, y)
    balanced = fit_model(make_model("nb", config.MODEL_FEATURES), X, y, balanced=True)
    assert not np.allclose(plain[-1].class_prior_, 0.25)
    assert np.allclose(balanced[-1].class_prior_, 0.25)


def test_params_are_set(featured):
    pipe = make_model("dt", config.MODEL_FEATURES, model__max_depth=3, balanced=True)
    assert pipe[-1].max_depth == 3


def test_unknown_kind():
    with pytest.raises(ValueError):
        make_model("svm", config.MODEL_FEATURES)

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src.evaluate import evaluate


def fitted_dummy(y):
    X = pd.DataFrame({"x": np.zeros(len(y))})
    return DummyClassifier(strategy="most_frequent").fit(X, y), X


def test_evaluate_most_frequent(tmp_path):
    y = pd.Series([1] * 5 + [2] * 80 + [3] * 10 + [4] * 5)
    model, X = fitted_dummy(y)
    m = evaluate(model, X, y, "dummy", results_dir=tmp_path)

    assert m["name"] == "dummy"
    assert m["accuracy"] == pytest.approx(0.80)
    # only class 2 gets an f1 above 0: 2 * 0.8 * 1 / 1.8
    assert m["macro_f1"] == pytest.approx((2 * 0.8 / 1.8) / 4)
    assert m["per_class"]["2"]["recall"] == 1.0
    assert m["per_class"]["1"]["support"] == 5
    assert m["confusion_matrix"][1] == [0, 80, 0, 0]


def test_evaluate_writes_json(tmp_path):
    y = pd.Series([1, 2, 2, 2, 3, 4])
    model, X = fitted_dummy(y)
    m = evaluate(model, X, y, "dummy", results_dir=tmp_path)
    saved = json.loads((tmp_path / "dummy.json").read_text())
    assert saved == m


def test_missing_class_still_4x4(tmp_path):
    y = pd.Series([2, 2, 3, 4, 2])  # no severity 1
    model, X = fitted_dummy(y)
    m = evaluate(model, X, y, "dummy", results_dir=tmp_path)
    assert len(m["confusion_matrix"]) == 4
    assert all(len(row) == 4 for row in m["confusion_matrix"])
    assert set(m["per_class"]) == {"1", "2", "3", "4"}

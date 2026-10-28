from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from src import config


# fill blanks, scale numbers, one-hot the categories
def build_preprocessor(features):
    cats = [c for c in features if c in config.CATEGORICAL_FEATURES]
    nums = [c for c in features if c not in config.CATEGORICAL_FEATURES]
    numeric = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),  # keep the column even if it's all NaN
        StandardScaler(),
    )
    categorical = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="Unknown"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),  # naive bayes needs dense
    )
    return ColumnTransformer([("num", numeric, nums), ("cat", categorical, cats)])


def make_model(kind, features, **params):
    if kind == "lr":
        model = LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED)
    elif kind == "nb":
        model = GaussianNB()
    elif kind == "dt":
        model = DecisionTreeClassifier(random_state=config.RANDOM_SEED)
    else:
        raise ValueError(f"unknown model kind: {kind}")
    pipe = Pipeline([("prep", build_preprocessor(features)), ("model", model)])
    # balanced isn't a pipeline setting, fit_model handles it
    pipe.set_params(**{k: v for k, v in params.items() if k != "balanced"})
    return pipe


# balanced = rare severities count more
def fit_model(pipe, X, y, balanced=False):
    if balanced:
        return pipe.fit(X, y, model__sample_weight=compute_sample_weight("balanced", y))
    return pipe.fit(X, y)

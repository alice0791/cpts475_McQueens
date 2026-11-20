from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import numpy as np

from src import config

NB_BINS = 10
NB_UNKNOWN = 99

# converts xgb severity from 0-3 to 1-4
class SeverityXGBClassifier(XGBClassifier):

    def fit(self, X, y, sample_weight=None, **kwargs):
        y = np.asarray(y)
        severity_classes = np.asarray(config.CLASSES)
        self._severity_classes = severity_classes
        encoded = np.searchsorted(severity_classes, y)
        valid = (
            (encoded >= 0)
            & (encoded < len(severity_classes))
            & (severity_classes[np.clip(encoded, 0, len(severity_classes) - 1)] == y)
        )
        if not np.all(valid):
            unexpected = sorted(set(y) - set(config.CLASSES))
            raise ValueError(f"unexpected severity labels: {unexpected}")
        return super().fit(X, encoded, sample_weight=sample_weight, **kwargs)

    def predict(self, X, **kwargs):
        encoded = super().predict(X, **kwargs)
        return self._severity_classes[encoded.astype(int)]


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
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    )
    return ColumnTransformer([("num", numeric, nums), ("cat", categorical, cats)])


# naive bayes needs every column as categories, so bin the weather numbers
def build_nb_preprocessor(features):
    binned = [c for c in features if c in config.NUMERIC_WEATHER]
    cats = [c for c in features if c in config.CATEGORICAL_FEATURES]
    other = [c for c in features if c not in binned + cats]
    bins = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        KBinsDiscretizer(n_bins=NB_BINS, encode="ordinal", strategy="quantile", random_state=config.RANDOM_SEED),
    )
    # unseen values go to 99
    codes = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=NB_UNKNOWN)
    return ColumnTransformer([
        ("bins", bins, binned),
        ("other", make_pipeline(SimpleImputer(strategy="most_frequent"), codes), other),
        ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="Unknown"), clone(codes)), cats),
    ])


def make_model(kind, features, **params):
    if kind == "lr":
        model = LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED)
    elif kind == "nb":
        model = CategoricalNB(min_categories=NB_UNKNOWN + 1)
    elif kind == "dt":
        model = DecisionTreeClassifier(random_state=config.RANDOM_SEED)
    elif kind == "knn":
        model = KNeighborsClassifier(n_jobs=-1)
    elif kind == "rf":
        model = RandomForestClassifier(random_state=config.RANDOM_SEED, n_jobs=-1)
    elif kind == "xgb":
        model = SeverityXGBClassifier(
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
            eval_metric="mlogloss",
            objective="multi:softprob",
            num_class=len(config.CLASSES),
        )
    else:
        raise ValueError(f"unknown model kind: {kind}")
    prep = build_nb_preprocessor(features) if kind == "nb" else build_preprocessor(features)
    pipe = Pipeline([("prep", prep), ("model", model)])
    # balanced isn't a pipeline setting, fit_model handles it
    pipe.set_params(**{k: v for k, v in params.items() if k != "balanced"})
    return pipe


# balanced = rare severities count more
def fit_model(pipe, X, y, balanced=False):
    if balanced:
        if isinstance(pipe[-1], KNeighborsClassifier):
            X, y = resample(
                X,
                y,
                replace=True,
                n_samples=len(y),
                random_state=config.RANDOM_SEED,
                stratify=y,
            )
            return pipe.fit(X, y)
        return pipe.fit(X, y, model__sample_weight=compute_sample_weight("balanced", y))
    return pipe.fit(X, y)

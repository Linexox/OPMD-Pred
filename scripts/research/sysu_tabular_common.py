"""Training-fold-only clinical tables and optional photo PCA features."""
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


HISTORY_NAMES = ["smoking", "smoking_years", "smoking_amount", "alcohol", "alcohol_years", "alcohol_amount",
                 "betel", "betel_years", "betel_amount"]


def table_present(data, ids):
    return np.asarray([
        bool(data.records[s].tct) or any(bool(data.base[f"{name}_mask"][data.index[s]].any())
                                       for name in ("history", "dna", "methylation")) for s in ids])


class TableBuilder:
    def __init__(self, mode):
        if mode not in ("table", "table_image"):
            raise ValueError("Unknown table mode")
        self.mode, self.pca = mode, None

    def eligible(self, data, ids):
        available = table_present(data, ids)
        if self.mode == "table_image":
            available |= np.asarray([bool(data.present[data.index[s]]) for s in ids])
        return available

    def fit(self, data, ids):
        self.fit_ids = list(ids)
        self.photo_fit_ids = []
        if self.mode == "table_image":
            self.photo_fit_ids = [s for s in ids if data.present[data.index[s]]]
            features = np.asarray([data.images["baseline"][data.index[s], 0].numpy() for s in self.photo_fit_ids])
            n = min(16, len(features) - 1, features.shape[1])
            if n < 1:
                raise ValueError("No training-fold photographs for PCA")
            self.pca = PCA(n_components=n, svd_solver="full").fit(features)
        return self

    def transform(self, data, ids, remove=None):
        columns = {name: [] for name in HISTORY_NAMES + ["TCT", "DNA_ratio", "DNA_DI_2_5", "methylation_total"]}
        for sid in ids:
            i = data.index[sid]
            for j, name in enumerate(HISTORY_NAMES):
                columns[name].append(float(data.base["history"][i, j]) if data.base["history_mask"][i, j] else np.nan)
            columns["TCT"].append(data.records[sid].tct if data.records[sid].tct and remove != "tct" else np.nan)
            for j, name in enumerate(("DNA_ratio", "DNA_DI_2_5")):
                columns[name].append(float(data.base["dna"][i, j]) if data.base["dna_mask"][i, j] and remove != "dna" else np.nan)
            columns["methylation_total"].append(float(data.base["methylation"][i, 0]) * 10
                     if data.base["methylation_mask"][i, 0] and remove != "methylation" else np.nan)
        frame = pd.DataFrame(columns)
        # Keep the categorical column categorical even when a counterfactual
        # hides TCT for every queried row.
        frame["TCT"] = pd.Series(columns["TCT"], dtype=object)
        if self.mode == "table_image":
            photo_present = np.asarray([bool(data.present[data.index[s]]) for s in ids])
            projected = np.full((len(ids), self.pca.n_components_), np.nan)
            if photo_present.any():
                x = np.asarray([data.images["baseline"][data.index[s], 0].numpy()
                                for s, yes in zip(ids, photo_present) if yes])
                projected[photo_present] = self.pca.transform(x)
            for j in range(self.pca.n_components_):
                frame[f"photo_PC{j+1}"] = projected[:, j]
            frame["photo_present"] = photo_present.astype(float)
        return frame

    def eligible_after_removal(self, data, ids, name):
        present = []
        for sid in ids:
            i = data.index[sid]
            flags = {m: bool(data.base[f"{m}_mask"][i].any()) for m in ("history", "dna", "methylation")}
            flags["tct"] = bool(data.records[sid].tct)
            flags["image"] = self.mode == "table_image" and bool(data.present[i])
            flags[name] = False
            present.append(any(flags.values()))
        return np.asarray(present)


def logistic_model(frame, seed):
    numeric = [c for c in frame.columns if c != "TCT"]
    prep = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                              ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), ["TCT"])])
    return Pipeline([("preprocess", prep), ("classifier", LogisticRegression(
        C=1., class_weight="balanced", max_iter=2000, solver="lbfgs", random_state=seed))])


@contextmanager
def offline_predictions():
    """A prediction run must not establish any external network connection."""
    def denied(*args, **kwargs):
        raise RuntimeError("Network connection attempted during offline patient processing")
    with patch("socket.socket.connect", denied), patch("socket.create_connection", denied):
        yield

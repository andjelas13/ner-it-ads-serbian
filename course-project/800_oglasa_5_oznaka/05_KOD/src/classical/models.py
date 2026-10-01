"""Reusable sequence preparation and four required classical model adapters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.common.constants import BIO_LABELS, LABEL_TO_ID, SEED
from src.data.prepare_ner_dataset import fields
from src.features.token_features import sequence_features
from src.common.tokenization import Token, bio_to_spans, spans_to_bio


@dataclass
class SequenceExample:
    ad_id: str
    field: str
    tokens: list[Token]
    labels: list[str]
    # Deterministic, label-free features may be shared by all CV fits.
    cached_features: list[dict[str, object]] | None = None


def feature_rows(sequence: SequenceExample) -> list[dict[str, object]]:
    """Return cached features when available, otherwise compute them once."""
    if sequence.cached_features is None:
        sequence.cached_features = sequence_features(sequence.tokens, sequence.field)
    return sequence.cached_features


def make_sequences(ads: list[dict[str, Any]]) -> list[SequenceExample]:
    output = []
    for ad in ads:
        for field, text in fields(ad):
            tokens, labels = spans_to_bio(text, ad.get("annotations", []), field)
            if tokens:
                output.append(SequenceExample(str(ad["id"]), field, tokens, labels))
    return output


def sequences_to_entities(sequences: list[SequenceExample], predictions: list[list[str]]) -> set[tuple[str, str, int, int, str]]:
    entities = set()
    for sequence, labels in zip(sequences, predictions):
        entities.update((sequence.ad_id, sequence.field, start, end, tag)
                        for start, end, tag in bio_to_spans(sequence.tokens, labels))
    return entities


class IndependentTokenModel:
    def __init__(self, name: str, params: dict[str, Any], runtime_options: dict[str, Any] | None = None):
        self.name, self.params = name, params
        self.runtime_options = runtime_options or {}
        self.vectorizer = DictVectorizer(sparse=True)
        self.model: Any = None

    def _build(self) -> Any:
        if self.name == "nb":
            return MultinomialNB(**self.params)
        if self.name == "svm":
            return LinearSVC(max_iter=int(self.runtime_options.get("svm_max_iter", 5000)),
                             tol=float(self.runtime_options.get("svm_tol", 1e-3)),
                             dual="auto", random_state=SEED, **self.params)
        if self.name == "xgb":
            try:
                from xgboost import XGBClassifier
            except ImportError as exc:
                raise RuntimeError("Install xgboost to run the XGBoost baseline") from exc
            return XGBClassifier(objective="multi:softprob", num_class=len(BIO_LABELS), tree_method="hist",
                                 device=str(self.runtime_options.get("xgb_device", "cuda")),
                                 subsample=.8, colsample_bytree=.8, random_state=SEED,
                                 n_jobs=int(self.runtime_options.get("xgb_n_jobs", -1)),
                                 eval_metric="mlogloss", **self.params)
        raise ValueError(self.name)

    def fit(self, sequences: list[SequenceExample]) -> "IndependentTokenModel":
        features = [f for seq in sequences for f in feature_rows(seq)]
        labels = [LABEL_TO_ID[label] for seq in sequences for label in seq.labels]
        matrix = self.vectorizer.fit_transform(features)
        self.model = self._build()
        self.model.fit(matrix, labels)
        return self

    def predict(self, sequences: list[SequenceExample]) -> list[list[str]]:
        sizes = [len(seq.tokens) for seq in sequences]
        features = [f for seq in sequences for f in feature_rows(seq)]
        flat = self.model.predict(self.vectorizer.transform(features)).tolist()
        labels = [BIO_LABELS[int(value)] for value in flat]
        cursor, output = 0, []
        for size in sizes:
            output.append(labels[cursor:cursor+size]); cursor += size
        return output


class CRFModel:
    def __init__(self, params: dict[str, Any], runtime_options: dict[str, Any] | None = None):
        self.params = params
        self.runtime_options = runtime_options or {}
        self.model: Any = None

    def fit(self, sequences: list[SequenceExample]) -> "CRFModel":
        try:
            import sklearn_crfsuite
        except ImportError as exc:
            raise RuntimeError("Install sklearn-crfsuite to run CRF") from exc
        self.model = sklearn_crfsuite.CRF(algorithm="lbfgs",
                                          max_iterations=int(self.runtime_options.get("crf_max_iterations", 100)),
                                          all_possible_transitions=True, **self.params)
        self.model.fit([feature_rows(s) for s in sequences], [s.labels for s in sequences])
        return self

    def predict(self, sequences: list[SequenceExample]) -> list[list[str]]:
        return self.model.predict([feature_rows(s) for s in sequences])


GRIDS = {
    "nb": [{"alpha": a, "fit_prior": p} for a in (.01, .1, .5, 1.0) for p in (True, False)],
    "svm": [{"C": c, "class_weight": w} for c in (.1, 1., 10.) for w in (None, "balanced")],
    "xgb": [{"n_estimators": n, "max_depth": d, "learning_rate": lr}
            for n in (100, 300) for d in (3, 6) for lr in (.05, .1)],
    "crf": [{"c1": c1, "c2": c2} for c1 in (0., .1, .5) for c2 in (.01, .1, 1.)],
}


def make_model(name: str, params: dict[str, Any], runtime_options: dict[str, Any] | None = None):
    return CRFModel(params, runtime_options) if name == "crf" else IndependentTokenModel(name, params, runtime_options)

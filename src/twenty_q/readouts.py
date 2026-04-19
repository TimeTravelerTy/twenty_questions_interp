"""Frozen readouts trained on calibration and applied to self-chosen (M2).

Three decoder flavors per layer (DECISIONS.md D-11):

1. **Nearest-centroid retrieval** — cheapest; robust to small N; preferred
   baseline per AxBench-style findings.
2. **Multinomial logistic regression** — slightly stronger but risks
   overfitting at tiny sample sizes.
3. **Binary attribute decoders** — one per question; the most data-efficient
   sanity check since each predicate has many positive and negative examples.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .banks import Bank


@dataclass
class NearestCentroidDecoder:
    class_ids: list[str]
    centroids: np.ndarray  # (n_classes, hidden_size)

    def predict(self, X: np.ndarray) -> list[str]:
        # Cosine similarity — scale-invariant and tends to work better than L2
        # for transformer hidden states.
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        Cn = self.centroids / (np.linalg.norm(self.centroids, axis=1, keepdims=True) + 1e-9)
        sims = Xn @ Cn.T  # (N, n_classes)
        idx = np.argmax(sims, axis=1)
        return [self.class_ids[i] for i in idx]


def fit_nearest_centroid(
    X: np.ndarray, y: list[str], class_ids: list[str] | None = None
) -> NearestCentroidDecoder:
    class_ids = class_ids or sorted(set(y))
    centroids = np.stack(
        [X[[i for i, yy in enumerate(y) if yy == c]].mean(axis=0) for c in class_ids]
    )
    return NearestCentroidDecoder(class_ids=class_ids, centroids=centroids)


def loo_accuracy_nearest_centroid(
    X: np.ndarray, y: list[str], class_ids: list[str]
) -> float:
    """Leave-one-run-out accuracy: each run is the test point, train on the rest."""
    n = len(y)
    correct = 0
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        dec = fit_nearest_centroid(X[mask], [y[j] for j in range(n) if mask[j]], class_ids)
        pred = dec.predict(X[i : i + 1])[0]
        correct += int(pred == y[i])
    return correct / n


def loo_accuracy_logreg(
    X: np.ndarray, y: list[str], class_ids: list[str], C: float = 1.0
) -> float:
    """Leave-one-run-out accuracy with multinomial logistic regression."""
    n = len(y)
    correct = 0
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        scaler = StandardScaler().fit(X[mask])
        X_train = scaler.transform(X[mask])
        X_test = scaler.transform(X[i : i + 1])
        clf = LogisticRegression(max_iter=2000, C=C, solver="lbfgs")
        clf.fit(X_train, [y[j] for j in range(n) if mask[j]])
        pred = clf.predict(X_test)[0]
        correct += int(pred == y[i])
    return correct / n


def attribute_labels(y_candidate: list[str], bank: Bank, question_id: str) -> list[int]:
    return [bank.answer(c, question_id) for c in y_candidate]


def loo_accuracy_binary(
    X: np.ndarray, y_binary: list[int], C: float = 1.0
) -> tuple[float, float]:
    """Return (LOO accuracy, majority-class baseline) for a binary logistic probe."""
    n = len(y_binary)
    yarr = np.array(y_binary)
    majority = max(yarr.mean(), 1 - yarr.mean())
    # Degenerate case: only one class present -> can't train.
    if len(set(y_binary)) < 2:
        return majority, majority
    correct = 0
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        scaler = StandardScaler().fit(X[mask])
        clf = LogisticRegression(max_iter=2000, C=C)
        clf.fit(scaler.transform(X[mask]), yarr[mask])
        pred = clf.predict(scaler.transform(X[i : i + 1]))[0]
        correct += int(pred == yarr[i])
    return correct / n, float(majority)


def layerwise_loo_accuracy_nearest_centroid(
    states: Sequence[torch.Tensor],
    labels: list[str],
    class_ids: list[str],
) -> list[float]:
    n_layers = states[0].shape[0]
    accs: list[float] = []
    for ell in range(n_layers):
        X = np.stack([t[ell].numpy() for t in states], axis=0)
        accs.append(loo_accuracy_nearest_centroid(X, labels, class_ids))
    return accs


def layerwise_cross_nearest_centroid(
    states_src: Sequence[torch.Tensor],
    labels_src: list[str],
    states_tgt: Sequence[torch.Tensor],
    labels_tgt: list[str],
    class_ids: list[str],
) -> list[float]:
    n_layers = states_src[0].shape[0]
    accs: list[float] = []
    for ell in range(n_layers):
        Xs = np.stack([t[ell].numpy() for t in states_src], axis=0)
        Xt = np.stack([t[ell].numpy() for t in states_tgt], axis=0)
        dec = fit_nearest_centroid(Xs, labels_src, class_ids)
        pred = dec.predict(Xt)
        correct = sum(1 for p, y in zip(pred, labels_tgt, strict=True) if p == y)
        accs.append(correct / len(labels_tgt))
    return accs


def within_between_contrast(
    states_by_class: dict[str, Sequence[torch.Tensor]],
) -> dict[str, Any]:
    """Layerwise within-vs-between cosine contrast plus a post-L13 summary."""
    class_ids = list(states_by_class.keys())
    within_pairs: list[torch.Tensor] = []
    between_pairs: list[torch.Tensor] = []
    for tensors in states_by_class.values():
        for i in range(len(tensors)):
            for j in range(i + 1, len(tensors)):
                within_pairs.append(
                    torch.nn.functional.cosine_similarity(
                        tensors[i], tensors[j], dim=1
                    )
                )
    for a_idx in range(len(class_ids)):
        for b_idx in range(a_idx + 1, len(class_ids)):
            for ta in states_by_class[class_ids[a_idx]]:
                for tb in states_by_class[class_ids[b_idx]]:
                    between_pairs.append(
                        torch.nn.functional.cosine_similarity(ta, tb, dim=1)
                    )

    result: dict[str, Any] = {}
    if within_pairs:
        within = torch.stack(within_pairs, dim=0).mean(dim=0)
        result["within_by_layer"] = [float(x) for x in within.tolist()]
    if between_pairs:
        between = torch.stack(between_pairs, dim=0).mean(dim=0)
        result["between_by_layer"] = [float(x) for x in between.tolist()]
    if "within_by_layer" in result and "between_by_layer" in result:
        contrast = [
            w - b
            for w, b in zip(
                result["within_by_layer"], result["between_by_layer"], strict=True
            )
        ]
        result["contrast_by_layer"] = contrast
        post13 = contrast[13:]
        result["contrast_post13"] = float(sum(post13) / max(1, len(post13)))
    return result

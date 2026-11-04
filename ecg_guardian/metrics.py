"""Failure-detection metrics. 'score' = higher means more likely wrong; 'error' = 1 if prediction wrong."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def risk_coverage(score: np.ndarray, error: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep the most-trusted records first. Returns (coverage, risk) after each kept record."""
    order = np.argsort(score, kind="stable")          # lowest suspicion kept first
    err = np.asarray(error, float)[order]
    n = np.arange(1, len(err) + 1)
    return n / len(err), np.cumsum(err) / n


def aurc(score: np.ndarray, error: np.ndarray) -> float:
    """Area under the risk-coverage curve (lower is better), mean of risk over all coverages."""
    _, risk = risk_coverage(score, error)
    return float(risk.mean())


def failure_auroc(score, error) -> float:
    return float(roc_auc_score(error, score)) if 0 < np.sum(error) < len(error) else float("nan")


def failure_prauc(score, error) -> float:
    return float(average_precision_score(error, score)) if np.sum(error) > 0 else float("nan")


def errors_caught_at(score, error, review_fraction: float) -> float:
    """Share of all errors that land in the top `review_fraction` most suspicious records."""
    error = np.asarray(error)
    k = int(np.ceil(review_fraction * len(error)))
    top = np.argsort(-np.asarray(score), kind="stable")[:k]
    return float(error[top].sum() / max(error.sum(), 1))


def bootstrap_diff(score_a, score_b, error, metric=aurc, n=2000, seed=0) -> tuple[float, float, float]:
    """metric(a) - metric(b) with a 95% bootstrap CI over records."""
    rng = np.random.default_rng(seed)
    a, b, e = map(np.asarray, (score_a, score_b, error))
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(e), len(e))
        if e[i].sum() == 0:
            continue
        diffs.append(metric(a[i], e[i]) - metric(b[i], e[i]))
    return float(metric(a, e) - metric(b, e)), *map(float, np.percentile(diffs, [2.5, 97.5]))

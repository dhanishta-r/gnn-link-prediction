"""Shared evaluation for Milestone 2.

Every model is scored by the same function, in two settings:

  easy   positives vs. random negatives (the Milestone 1 style of test)
  heart  each positive vs. its own K hard negatives from HeaRT

Metrics
  AUC, AP     positives against all negatives pooled (as in Milestone 1)
  MRR         mean reciprocal rank of each positive among its negatives
  Hits@K      share of positives ranked in the top K (K = 1, 3, 10, 100)

In the easy setting every positive is ranked against the whole set of random
negatives; in the heart setting against its own K hard negatives. Ties get the
average of the best and worst possible rank. This is the same ranking rule as
HeaRT's `eval_mrr` (benchmarking/evalutors.py).

Usage
-----
    from evaluate import evaluate_model, append_results

    def score_fn(edges):          # edges: (N, 2) numpy array of node pairs
        return scores             # (N,) numpy array or torch tensor, higher = more likely a link

    rows = evaluate_model(score_fn, data, split="valid")   # tuning
    rows = evaluate_model(score_fn, data, split="test")    # final numbers only
    append_results("results/m2/raw_baselines.csv", rows, dataset=data.name,
                   model="GAE", config="lr=0.01,hidden=32", seed=0, member="C")
"""

from __future__ import annotations

import csv
import os

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

K_LIST = (1, 3, 10, 100)
METRICS = ["AUC", "AP", "MRR"] + [f"Hits@{k}" for k in K_LIST]
CSV_COLUMNS = ["dataset", "model", "config", "seed", "setting", "split"] + METRICS + ["member", "notes"]


def _to_numpy(a) -> np.ndarray:
    if hasattr(a, "detach"):          # torch tensor
        a = a.detach().cpu().numpy()
    return np.asarray(a, dtype=np.float64).reshape(-1)


def ranks(pos_scores: np.ndarray, neg_scores: np.ndarray) -> np.ndarray:
    """Rank of each positive among its negatives (1 = best).

    pos_scores: (P,)    neg_scores: (P, K) or (K,) shared by all positives.
    """
    pos = np.asarray(pos_scores, dtype=np.float64).reshape(-1)
    neg = np.asarray(neg_scores, dtype=np.float64)
    if neg.ndim == 1:                              # one shared negative set
        s = np.sort(neg)
        optimistic = len(s) - np.searchsorted(s, pos, side="left")    # neg >= pos
        pessimistic = len(s) - np.searchsorted(s, pos, side="right")  # neg >  pos
    else:
        optimistic = (neg >= pos[:, None]).sum(axis=1)
        pessimistic = (neg > pos[:, None]).sum(axis=1)
    return 0.5 * (optimistic + pessimistic) + 1.0


def compute_metrics(pos_scores, neg_scores) -> dict:
    """All metrics for one setting. neg_scores is (K,) shared or (P, K) per positive."""
    pos = np.asarray(pos_scores, dtype=np.float64).reshape(-1)
    neg = np.asarray(neg_scores, dtype=np.float64)
    if not (np.isfinite(pos).all() and np.isfinite(neg).all()):
        raise ValueError("Scores contain NaN or infinity")
    y_true = np.concatenate([np.ones(len(pos)), np.zeros(neg.size)])
    y_score = np.concatenate([pos, neg.reshape(-1)])
    r = ranks(pos, neg)
    out = {
        "AUC": 100 * roc_auc_score(y_true, y_score),
        "AP": 100 * average_precision_score(y_true, y_score),
        "MRR": 100 * float((1.0 / r).mean()),
    }
    for k in K_LIST:
        out[f"Hits@{k}"] = 100 * float((r <= k).mean())
    return out


def _score_in_chunks(score_fn, edges: np.ndarray, chunk: int) -> np.ndarray:
    parts = [_to_numpy(score_fn(edges[i:i + chunk])) for i in range(0, len(edges), chunk)]
    out = np.concatenate(parts) if parts else np.zeros(0)
    if len(out) != len(edges):
        raise ValueError(f"score_fn returned {len(out)} scores for {len(edges)} edges")
    return out


def evaluate_model(score_fn, data, split: str = "test", chunk: int = 200_000) -> list:
    """Score a model in both settings. Returns one result row (dict) per setting.

    score_fn(edges) takes an (N, 2) integer numpy array of node pairs and
    returns N scores, higher meaning more likely to be a link. All values are
    in percent.
    """
    if split not in ("valid", "test"):
        raise ValueError("split must be 'valid' or 'test'")
    pos = data.pos(split)
    pos_scores = _score_in_chunks(score_fn, pos, chunk)

    easy_scores = _score_in_chunks(score_fn, data.easy_neg[split], chunk)

    hard = data.hard_neg[split]                       # (P, K, 2)
    hard_scores = _score_in_chunks(score_fn, hard.reshape(-1, 2), chunk).reshape(hard.shape[0], hard.shape[1])

    rows = []
    for setting, neg in (("easy", easy_scores), ("heart", hard_scores)):
        row = {"setting": setting, "split": split}
        row.update(compute_metrics(pos_scores, neg))
        rows.append(row)
    return rows


def append_results(path: str, rows: list, dataset: str, model: str, config: str = "default",
                   seed: int = 0, member: str = "", notes: str = "") -> None:
    """Append result rows to the shared CSV (creates it with a header if needed)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new_file:
            writer.writeheader()
        for row in rows:
            full = {"dataset": dataset, "model": model, "config": config, "seed": seed,
                    "member": member, "notes": notes}
            full.update({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
            writer.writerow(full)


def format_rows(rows: list) -> str:
    """Small text table for printing in a notebook."""
    lines = [f"{'setting':8s} {'split':6s} " + " ".join(f"{m:>8s}" for m in METRICS)]
    for row in rows:
        lines.append(f"{row['setting']:8s} {row['split']:6s} " + " ".join(f"{row[m]:8.2f}" for m in METRICS))
    return "\n".join(lines)

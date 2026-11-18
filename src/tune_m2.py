"""Milestone 2: hyper-parameter tuning of GAE and VGAE on the HeaRT split.

Protocol (same effort as SEAL: 6 configurations per model and dataset)
  1. Every configuration in GRID is trained once (seed 0) and scored on the
     VALIDATION edges only.
  2. For each dataset, model and setting, the configuration with the best
     validation score is chosen (AUC for easy negatives, MRR for HeaRT
     negatives). The test edges are not used for this choice.
  3. Each chosen configuration is re-run with seeds 0-9 and scored on the
     validation and test edges.

All rows go to results/m2/raw_tuning.csv in the shared format, so
src/aggregate_m2.py can merge them with the baselines and SEAL.

Command line
    python src/tune_m2.py --datasets cora citeseer pubmed
"""

from __future__ import annotations

import argparse
import os
import time

from baselines_m2 import train_autoencoder
from evaluate import append_results, evaluate_model
from heart_data import check_no_leak, load_heart

DEFAULT_OUT = os.path.join("results", "m2", "raw_tuning.csv")

# hidden = first GCN layer, out = embedding size, wd = weight decay
GRID = [
    dict(hidden=16, out=8, lr=0.001, epochs=400, wd=1e-4),
    dict(hidden=32, out=16, lr=0.001, epochs=400, wd=1e-4),
    dict(hidden=32, out=16, lr=0.005, epochs=400, wd=1e-4),
    dict(hidden=32, out=16, lr=0.01, epochs=200, wd=1e-4),
    dict(hidden=32, out=16, lr=0.01, epochs=400, wd=1e-4),
    dict(hidden=64, out=32, lr=0.01, epochs=400, wd=1e-4),
]
SELECT_BY = {"easy": "AUC", "heart": "MRR"}


def config_name(c: dict) -> str:
    return f"hidden={c['hidden']},out={c['out']},lr={c['lr']},epochs={c['epochs']},wd={c['wd']}"


def _train(data, variational, c, seed):
    return train_autoencoder(data, variational=variational, seed=seed, epochs=c["epochs"], lr=c["lr"],
                             hidden_channels=c["hidden"], out_channels=c["out"], weight_decay=c["wd"])


def tune(datasets=("cora", "citeseer", "pubmed"), seeds=range(10), out_csv: str = DEFAULT_OUT,
         root: str = "data", grid=GRID, member: str = "B"):
    for name in datasets:
        data = load_heart(name, root=root)
        check_no_leak(data, verbose=False)
        print(data.summary())
        for model, variational in (("GAE", False), ("VGAE", True)):
            # step 1: one seed per configuration, validation only
            scores = {}
            for c in grid:
                start = time.time()
                rows = evaluate_model(_train(data, variational, c, seed=0), data, split="valid")
                append_results(out_csv, rows, dataset=name, model=model, config=config_name(c), seed=0,
                               member=member, notes="tuning")
                scores[config_name(c)] = {r["setting"]: r[SELECT_BY[r["setting"]]] for r in rows}
                print(f"  {model:4s} {config_name(c):45s} valid: easy AUC {scores[config_name(c)]['easy']:.2f} | "
                      f"heart MRR {scores[config_name(c)]['heart']:.2f}  ({time.time() - start:.0f}s)")

            # step 2: choose on validation
            chosen = {s: max(scores, key=lambda k: scores[k][s]) for s in SELECT_BY}
            for s, cfg in chosen.items():
                print(f"  -> {model} best for {s} ({SELECT_BY[s]} on validation): {cfg}")

            # step 3: 10 seeds for each chosen configuration
            by_name = {config_name(c): c for c in grid}
            seeds = list(seeds)
            for cfg in dict.fromkeys(chosen.values()):
                for seed in seeds:
                    score = _train(data, variational, by_name[cfg], seed)
                    for split in ("valid", "test"):
                        append_results(out_csv, evaluate_model(score, data, split=split), dataset=name,
                                       model=model, config=cfg, seed=seed, member=member, notes="final_10seeds")
                print(f"  {model} {cfg}: {len(seeds)} seeds done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["cora", "citeseer", "pubmed"])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--root", default="data")
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()
    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} already exists. Delete or rename it first.")
    tune(args.datasets, range(args.seeds), args.out, args.root)

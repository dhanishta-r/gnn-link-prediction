# Milestone 2 – team handover

Deadline: **Wednesday 18 November 2026, 23:59** (Moodle, GitHub repository link).
Please send your results file by **Tuesday evening**, so the final table and README can be finished on Wednesday.

## Who does what

| Part | Task | Results file |
|---|---|---|
| A | **SEAL**: run the PyTorch Geometric SEAL example on Cora, CiteSeer and PubMed, adapted to the shared split, and tune it | `results/m2/raw_seal.csv` |
| B | **Tuning GAE/VGAE and the heuristics** with the same effort (GAE/VGAE: learning rate, hidden size, epochs, dropout; heuristics: Common Neighbours, Adamic–Adar, Resource Allocation, Katz and its parameter) | `results/m2/raw_tuning.csv` |
| C | **Shared setup, baselines, final table, README** (done: loader, evaluator, baseline runner, merge script) | `results/m2/raw_baselines.csv` |

Each part writes to its **own** file, so we never edit the same CSV and get no merge conflicts.

## The shared setup (please use it, do not load the data yourself)

```python
import sys; sys.path.insert(0, "src")
from heart_data import load_heart, check_no_leak
from evaluate import evaluate_model, append_results

data = load_heart("cora")              # or "citeseer", "pubmed"; downloads HeaRT on first use
check_no_leak(data)
x, edge_index = data.to_torch()        # node features and training graph for PyTorch Geometric
# data.train_pos  (E, 2) numpy array of training edges
# data.adj        scipy sparse adjacency matrix of the training edges (for heuristics)

def score_fn(edges):                   # (N, 2) numpy array of node pairs -> N scores, higher = more likely a link
    ...

rows = evaluate_model(score_fn, data, split="valid")    # for tuning
rows = evaluate_model(score_fn, data, split="test")     # only for configurations you report
append_results("results/m2/raw_seal.csv", rows, dataset=data.name,
               model="SEAL", config="hops=2,hidden=32", seed=0, member="A")
```

A full working example is in `notebooks/m2_shared_setup_test.ipynb`. Part B can reuse
`train_autoencoder` from `src/baselines_m2.py`: it takes the learning rate, sizes, dropout,
weight decay and epochs as arguments.

## Rules

1. **Train on `data.train_pos` only.** Never add validation or test edges to the graph or to the training labels.
2. **Tune on `split="valid"` only.** Run `split="test"` for the configurations you want reported, not to choose between them. The merge script picks the configuration with the best validation score (AUC on easy negatives, MRR on HeaRT negatives).
3. **Same effort for every model:** 12 configurations per model and dataset, one seed each while tuning. Then run the chosen configuration with **seeds 0–9** and write both the valid and the test rows.
4. **Record everything**, including configurations that did not help. The assignment asks for unsuccessful experiments too.
5. **Naming:** use the model names `GAE`, `VGAE`, `SEAL`, `CN`, `AA`, `RA`, `Katz`. Use `config="default"` for an untuned run and a short description such as `lr=0.005,hidden=64` otherwise. The same configuration must always have exactly the same `config` text.
6. Add a short note (5–10 lines) on what you changed, why you expected it to help, and what happened. It goes into the README.

## Things to know

- **One fixed split.** HeaRT has a single train/valid/test split (85/5/10%). Our 10 runs are now 10 random seeds on that split, not 10 random splits as in Milestone 1.
- **Two settings.** `easy`: each positive against random negatives (the Milestone 1 style). `heart`: each positive against its own 500 hard negatives. AUC and AP are the main metrics in the easy setting, MRR and Hits@10 in the HeaRT setting.
- **A few hard negatives are true edges.** 0.1–0.4% of HeaRT's hard negatives are other held-out positives, because HeaRT only filters the edges known at that stage. We keep them as shipped, so our numbers stay comparable with published ones. `check_no_leak` reports them under "allowed".
- **Check of the pipeline:** Common Neighbours and Adamic–Adar on Cora give 9.78 and 11.91 MRR in the HeaRT setting, the same as published HeaRT-setting results.
- **SEAL and negatives.** SEAL scores a pair from its enclosing subgraph, so scoring 500 negatives per positive is slow, mainly on PubMed. Start PubMed early and use a GPU. If it does not finish, report what was run and say so.

## Building the final table

```bash
python src/aggregate_m2.py
```

It reads every `results/m2/raw_*.csv` and writes `results/m2/summary_results.csv` and
`results/m2/results_table.md` (the table for the README).

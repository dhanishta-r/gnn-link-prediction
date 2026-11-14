<!--
DRAFT of the Milestone 2 section for README.md.
Parts marked TODO need the results and the notes from parts A and B.
When complete, paste this section into README.md after "Milestone 1 at a glance",
tick Milestone 2 in the Roadmap, and add the new files to "Files in the repository".
-->

## Milestone 2: Improve

Milestone 1 compared GAE/VGAE with simple heuristics on easy random negatives and untuned models. Milestone 2 makes that comparison fair in three ways: a stronger GNN (SEAL), harder test examples (HeaRT), and equal tuning effort for every model, including the heuristics.

### What we changed and why

| Change | Why we expected it to matter |
|---|---|
| **Hard negatives from HeaRT** (500 per positive, chosen to be close to the positive edge) alongside the easy random negatives | Random negative pairs almost never share a neighbour, so almost any method separates them. Hard negatives test whether a model can rank the true link above plausible alternatives. |
| **Ranking metrics** (MRR, Hits@K) in addition to AUC and AP | AUC is dominated by the many easy pairs. MRR asks how high the true link is ranked among its own negatives. |
| **One fixed split for all models** (HeaRT's 85/5/10% split) with a shared loader and evaluator | In Milestone 1 the split changed from run to run. Now every model trains on the same graph and is tested on the same pairs, so differences come from the model, not the split. |
| **SEAL** | TODO (part A): 2–3 sentences. SEAL classifies the subgraph around a node pair, so it can learn structural patterns such as common neighbours that GAE's node embeddings cannot express. |
| **Equal tuning of GAE, VGAE and the heuristics** | TODO (part B): 2–3 sentences. HeaRT reports that baselines are often under-tuned; Milestone 1 used default settings only. |

### Experimental protocol

- **Data:** Cora, CiteSeer and PubMed with the fixed HeaRT split. Models see the training edges only.
- **Model selection:** hyper-parameters are chosen on the validation edges only (AUC in the easy setting, MRR in the HeaRT setting). The test edges are used once, for the reported numbers.
- **Budget:** 12 configurations per model and dataset. TODO: confirm with A and B.
- **Runs:** 10 random seeds (0–9) per reported configuration, mean ± standard deviation. The heuristics have no randomness, so they have one run.
- **Leak check:** `check_no_leak` confirms that no validation or test edge is in the training graph and that no negative is a training edge.

Because the split is now fixed, the 10 runs are 10 seeds on one split, not 10 random splits as in Milestone 1. The Milestone 1 numbers are therefore not directly comparable; the "default" rows below are the Milestone 1 models re-run on the HeaRT split.

### Results

TODO: paste `results/m2/results_table.md` here once all three result files are in.

All individual runs are in `results/m2/raw_baselines.csv`, `raw_seal.csv` and `raw_tuning.csv`; means and standard deviations for every configuration are in `results/m2/summary_results.csv`.

### What worked, what did not

TODO after the results are in. Points to cover:

- How much each model drops from easy to hard negatives.
- Whether tuning changed the ranking of GNNs and heuristics.
- Whether SEAL beats GAE/VGAE, and at what computational cost.
- Changes that did not help, with possible reasons.

### Limitations

- **One split.** Results on a single fixed split do not show split-to-split variation; the standard deviations reflect random seeds only.
- **Some hard negatives are true edges.** 0.1–0.4% of HeaRT's hard negatives are other held-out positives, because HeaRT only filters the edges known at that stage. We keep them as shipped so that our numbers are comparable with published results.
- **Heuristic ties.** Common-neighbour scores are 0 for many pairs. Tied scores get the average of the best and worst rank, as in HeaRT's own evaluation code.
- TODO: SEAL runtime limits (part A), anything not finished.

### How to reproduce Milestone 2

```bash
pip install -r requirements.txt
python src/baselines_m2.py          # default GAE, VGAE, CN, AA on the HeaRT split (downloads HeaRT on first use)
# TODO (part A): SEAL command
# TODO (part B): tuning command
python src/aggregate_m2.py          # builds results/m2/summary_results.csv and results_table.md
```

Or open `notebooks/m2_baselines.ipynb` in Google Colab and choose Runtime → Run all.

### New files

| File | Purpose |
|---|---|
| `src/heart_data.py` | Shared loader: HeaRT split, easy and hard negatives, leak check |
| `src/evaluate.py` | Shared evaluator: AUC, AP, MRR, Hits@K in both settings; results CSV writer |
| `src/baselines_m2.py` | Default GAE, VGAE, Common Neighbours and Adamic–Adar on the HeaRT split |
| `src/aggregate_m2.py` | Merges all result files into the final table |
| `notebooks/m2_shared_setup_test.ipynb` | Test of the loader and evaluator on Cora |
| `notebooks/m2_baselines.ipynb` | Runs the baselines and shows the table |
| `results/m2/` | Raw runs, summary and final table |
| `docs/milestone2_handover.md` | Task split and rules for the team |

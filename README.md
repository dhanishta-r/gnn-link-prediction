# Do Graph Neural Networks Really Beat Counting Common Neighbours?

Group project for the Deep Learning course at BME (Budapest University of Technology and Economics).

## Team
**Team name:** TCM Dynamics

| Name | Neptune code |
|---|---|
| Song Ma | SAGUWI |
| İlknur Pehlivan | JE5XG1  |
| Bhuwaneshwaree Ragavan | EWX1Q5 |

## Project description

**Link prediction** means guessing which missing connections exist in a network, as used in friend and product recommendation, citation suggestion and drug-interaction discovery. Many papers report that graph neural networks (GNNs) beat simple rules, but the [HeaRT study (NeurIPS 2023)](https://arxiv.org/abs/2306.10453) found problems in how these comparisons were made: baselines performing below their real level, inconsistent data splits, and test examples that were too easy.

Our question: **when do GNNs really win?**

This repository contains **Milestone 1: Reproduce** and **Milestone 2: Improve**.

---

## Milestone 1 at a glance

We reproduced the Graph Autoencoder (GAE) and Variational Graph Autoencoder (VGAE) from [Kipf & Welling (2016)](https://arxiv.org/abs/1611.07308) using the official PyTorch Geometric example, and compared them with two non-neural baselines, Common Neighbors and Adamic–Adar, on **exactly the same hidden edges**.

- **GAE is reproduced on all three datasets.** On Cora it reaches 90.69 ± 0.93% AUC, against 91.0% in the paper and 90.6 ± 0.9% in an independent re-run of the same code.
- **VGAE is reproduced on PubMed only.** It falls 2.3 AUC points short on Cora and 4.2 on CiteSeer.
- **Both GNNs beat the heuristics by 18–32 AUC points.** GAE beat Adamic–Adar in all 30 paired runs.

### Final results

Mean ± standard deviation over 10 random edge splits, in %. All models are evaluated on the same split in each run and trained for 400 epochs.

| Dataset | Model | AUC (%) | AP (%) | Paper AUC / AP | Reproduced? |
|---|---|---|---|---|---|
| Cora | GAE | 90.69 ± 0.93 | 91.25 ± 1.13 | 91.0 / 92.0 | Yes |
| Cora | VGAE | 89.05 ± 1.08 | 89.48 ± 0.92 | 91.4 / 92.6 | No (−2.3 / −3.1) |
| Cora | Common Neighbors | 71.92 ± 0.92 | 71.69 ± 0.98 | — | — |
| Cora | Adamic–Adar | 71.98 ± 0.93 | 72.01 ± 0.97 | — | — |
| CiteSeer | GAE | 88.82 ± 1.12 | 89.63 ± 1.30 | 89.5 / 89.9 | Yes |
| CiteSeer | VGAE | 86.61 ± 1.67 | 87.25 ± 1.87 | 90.8 / 92.0 | No (−4.2 / −4.7) |
| CiteSeer | Common Neighbors | 66.49 ± 0.92 | 66.42 ± 0.97 | — | — |
| CiteSeer | Adamic–Adar | 66.50 ± 0.91 | 66.50 ± 0.87 | — | — |
| PubMed | GAE | 96.60 ± 0.18 | 96.68 ± 0.20 | 96.4 / 96.5 | Yes |
| PubMed | VGAE | 95.08 ± 0.59 | 95.21 ± 0.53 | 94.4 / 94.7 | Yes |
| PubMed | Common Neighbors | 64.29 ± 0.38 | 64.25 ± 0.38 | — | — |
| PubMed | Adamic–Adar | 64.29 ± 0.38 | 64.29 ± 0.40 | — | — |

"Reproduced" means both AUC and AP lie within max(1 point, 1 standard deviation) of Table 1 in the paper. All 120 individual runs are in [`results/raw_results.csv`](results/raw_results.csv).

---

## Milestone 2: Improve

Milestone 1 compared GAE/VGAE with simple heuristics on **easy random negatives** and **untuned** models, which is the setting the HeaRT study criticises. Milestone 2 makes the comparison fairer in three ways: harder test examples (HeaRT), a stronger GNN (SEAL), and tuning with the same budget.

### What we changed and why

| Change | Why we expected it to matter |
|---|---|
| **Hard negatives from HeaRT**: each validation/test positive is ranked against its own 500 hard negatives, chosen by heuristics to be close to the positive edge | Random negative pairs almost never share a neighbour, so almost any method separates them. Hard negatives test whether a model can rank the true link above plausible alternatives. |
| **Ranking metrics** (MRR, Hits@K) alongside AUC and AP | AUC is dominated by the many easy pairs. MRR measures how high the true link is ranked among its own candidates. |
| **One fixed split and one shared evaluator for every model** (HeaRT's 85/5/10% split; `src/heart_data.py`, `src/evaluate.py`) | In Milestone 1 each model run drew its own random split. Now all models train on the same graph and are scored on the same pairs, so differences come from the model, not the split. |
| **SEAL** (Zhang & Chen, 2018), PyG example adapted to the shared split | SEAL classifies the subgraph around a node pair with node labels (DRNL), so it can learn structural patterns such as common neighbours that GAE's node embeddings struggle to express. This is exactly the signal the heuristics use. |
| **Hyper-parameter tuning of GAE and VGAE** with the same budget as SEAL (6 configurations per model and dataset) | HeaRT reports that baselines are often under-tuned; Milestone 1 used the defaults only. Learning rate, embedding size and training length change how well the embeddings fit the training graph. |

### Experimental protocol

- **Data:** Cora, CiteSeer and PubMed with the fixed HeaRT split. Models see the training edges only (as graph structure and as training labels).
- **Model selection:** hyper-parameters are chosen on the **validation** edges only. The test edges are used only for the reported numbers.
- **Budget:** 6 configurations per model and dataset, one seed each, scored on validation. The chosen configuration is then run with **10 seeds (0–9)**; we report mean ± standard deviation on the test edges.
- **Heuristics** (Common Neighbors, Adamic–Adar) have no parameters and no randomness, so they have one run.
- **Leak check:** `check_no_leak` confirms that no validation or test edge is in the training graph and that no negative is a training edge.
- **Check of the pipeline:** our Common Neighbors and Adamic–Adar results in the HeaRT setting match published values exactly (for example 9.78 and 11.91 MRR on Cora).

Because the split is now fixed, the 10 runs are 10 seeds on one split, not 10 random splits as in Milestone 1. The "default" rows below are the Milestone 1 models re-run unchanged on the HeaRT split, which is our baseline.

### Results

Test results, mean ± standard deviation over seeds, in %. Easy = random negatives; HeaRT = 500 hard negatives per positive.

| Dataset | Model | Version | Easy AUC | Easy AP | HeaRT MRR | HeaRT Hits@10 | Seeds |
|---|---|---|---|---|---|---|---|
| Cora | CN | default | 70.85 ± 0.00 | 70.59 ± 0.00 | 9.78 ± 0.00 | 20.11 ± 0.00 | 1 |
| Cora | AA | default | 70.96 ± 0.00 | 71.16 ± 0.00 | 11.91 ± 0.00 | 24.10 ± 0.00 | 1 |
| Cora | GAE | default | 90.64 ± 0.49 | 90.87 ± 0.43 | 5.69 ± 0.34 | 12.26 ± 0.96 | 10 |
| Cora | GAE | tuned | 87.30 ± 1.26 | 87.21 ± 1.37 | 2.97 ± 0.23 | 5.12 ± 0.75 | 10 |
| Cora | VGAE | default | 88.72 ± 0.66 | 88.97 ± 0.63 | 4.19 ± 0.30 | 9.30 ± 1.50 | 10 |
| Cora | VGAE | tuned | 85.47 ± 2.18 | 85.24 ± 2.19 | 2.72 ± 0.41 | 4.82 ± 0.79 | 10 |
| Cora | SEAL | tuned | 80.42 ± 1.42 | 84.22 ± 0.76 | 10.94 ± 1.09 | 22.35 ± 1.78 | 10 |
| CiteSeer | CN | default | 67.49 ± 0.00 | 67.45 ± 0.00 | 8.42 ± 0.00 | 18.68 ± 0.00 | 1 |
| CiteSeer | AA | default | 67.49 ± 0.00 | 67.48 ± 0.00 | 10.82 ± 0.00 | 22.20 ± 0.00 | 1 |
| CiteSeer | GAE | default | 89.27 ± 0.94 | 90.46 ± 0.83 | 7.24 ± 0.69 | 18.66 ± 1.49 | 10 |
| CiteSeer | GAE | tuned | 81.43 ± 0.11 | 82.15 ± 0.12 | 1.86 ± 0.52 | 2.09 ± 1.22 | 10 |
| CiteSeer | VGAE | default | 87.36 ± 1.40 | 88.31 ± 1.38 | 4.72 ± 0.86 | 11.03 ± 3.48 | 10 |
| CiteSeer | VGAE | tuned | 79.64 ± 2.68 | 80.61 ± 2.40 | 1.87 ± 0.44 | 2.37 ± 1.23 | 10 |
| CiteSeer | SEAL | tuned | 79.95 ± 1.80 | 83.20 ± 1.64 | 8.71 ± 1.13 | 18.66 ± 0.95 | 10 |
| PubMed | CN | default | 63.90 ± 0.00 | 63.87 ± 0.00 | 2.28 ± 0.00 | 4.78 ± 0.00 | 1 |
| PubMed | AA | default | 63.90 ± 0.00 | 63.90 ± 0.00 | 2.63 ± 0.00 | 5.51 ± 0.00 | 1 |
| PubMed | GAE | default | 96.54 ± 0.19 | 96.78 ± 0.17 | 2.82 ± 0.14 | 3.78 ± 0.46 | 10 |
| PubMed | GAE | tuned | 95.12 ± 0.28 | 95.38 ± 0.26 | 2.42 ± 0.08 | 2.16 ± 0.28 | 10 |
| PubMed | VGAE | default | 94.84 ± 0.37 | 95.11 ± 0.33 | 2.46 ± 0.08 | 2.80 ± 0.21 | 10 |
| PubMed | VGAE | tuned | 94.78 ± 0.13 | 95.03 ± 0.11 | 2.41 ± 0.03 | 2.41 ± 0.07 | 10 |

Tuned configurations, chosen on the validation set only (AUC in the easy setting, MRR in the HeaRT setting):

| Dataset | Model | Setting | Chosen configuration | Validation score |
|---|---|---|---|---|
| Cora | GAE | easy | `hidden=64,out=32,lr=0.01,epochs=400,wd=0.0001` | 85.97 (AUC) |
| Cora | GAE | heart | `hidden=64,out=32,lr=0.01,epochs=400,wd=0.0001` | 3.21 (MRR) |
| Cora | VGAE | easy | `hidden=64,out=32,lr=0.01,epochs=400,wd=0.0001` | 84.81 (AUC) |
| Cora | VGAE | heart | `hidden=64,out=32,lr=0.01,epochs=400,wd=0.0001` | 3.03 (MRR) |
| CiteSeer | GAE | easy | `hidden=16,out=8,lr=0.001,epochs=400,wd=0.0001` | 80.42 (AUC) |
| CiteSeer | GAE | heart | `hidden=64,out=32,lr=0.01,epochs=400,wd=0.0001` | 1.81 (MRR) |
| CiteSeer | VGAE | easy | `hidden=64,out=32,lr=0.01,epochs=400,wd=0.0001` | 79.39 (AUC) |
| CiteSeer | VGAE | heart | `hidden=64,out=32,lr=0.01,epochs=400,wd=0.0001` | 1.81 (MRR) |
| PubMed | GAE | easy | `hidden=64,out=32,lr=0.01,epochs=400,wd=0.0001` | 95.10 (AUC) |
| PubMed | GAE | heart | `hidden=64,out=32,lr=0.01,epochs=400,wd=0.0001` | 2.24 (MRR) |
| PubMed | VGAE | easy | `hidden=64,out=32,lr=0.01,epochs=400,wd=0.0001` | 94.78 (AUC) |
| PubMed | VGAE | heart | `hidden=64,out=32,lr=0.01,epochs=400,wd=0.0001` | 2.24 (MRR) |
| Cora | SEAL | easy | `hops=1,hidden=32,lr=0.001,drop=0.5,ep=15` | 80.29 (AUC) |
| Cora | SEAL | heart | `hops=1,hidden=32,lr=0.001,drop=0.5,ep=15` | 12.77 (MRR) |
| CiteSeer | SEAL | easy | `hops=1,hidden=32,lr=0.0005,drop=0.5,ep=15` | 79.51 (AUC) |
| CiteSeer | SEAL | heart | `hops=1,hidden=32,lr=0.0005,drop=0.5,ep=15` | 9.13 (MRR) |

"default" = untuned settings from Milestone 1; "tuned" = best of 6 configurations chosen on the validation edges, chosen separately for the easy setting (by AUC) and the HeaRT setting (by MRR). SEAL has no default row because it was only run with tuning. SEAL on PubMed did not finish (see Limitations).

Metrics: **Easy AUC / AP** rank each test positive against random negatives (the Milestone 1 setting). **HeaRT MRR / Hits@10** rank each test positive against its own 500 hard negatives. All values in %. MRR, Hits@1/3/100 and the validation scores for every configuration are in [`results/m2/summary_results.csv`](results/m2/summary_results.csv); every single run, including all tuning runs, is in `results/m2/raw_baselines.csv`, `raw_seal.csv` and `raw_tuning.csv`.

### What worked, what did not

**Hard negatives reverse the ranking on Cora and CiteSeer.** With easy random negatives, GAE beats Adamic–Adar by about 20 AUC points, as in Milestone 1. With HeaRT's hard negatives, Adamic–Adar has about twice GAE's MRR on Cora (11.91 vs 5.69) and also beats it on CiteSeer (10.82 vs 7.24). GAE's high AUC mostly reflects that random pairs are easy to reject; when the negatives are nodes close to the positive pair, its inner-product embeddings rank many of them above the true link.

**SEAL helps where GAE fails, but does not beat the best heuristic.** SEAL roughly doubles GAE's HeaRT MRR on Cora (10.94 vs 5.69) and improves it on CiteSeer (8.71 vs 7.24), which fits the motivation: DRNL node labels let it use the local structure that the heuristics use. It is still 1–2 MRR points below Adamic–Adar, and on easy negatives it is about 10 AUC points below GAE (80.42 vs 90.64 on Cora). With only 15 epochs and 1-hop subgraphs, SEAL is probably under-trained.

**Tuning GAE and VGAE did not help: every tuned configuration is worse than the default.** On all three datasets and for both models, the best of the 6 configurations scores below the untuned default on the validation edges, and the test results confirm it. On Cora, tuned GAE falls from 90.64 to 87.30 easy AUC and from 5.69 to 2.97 HeaRT MRR; on CiteSeer the HeaRT MRR drops from 7.24 to 1.86. The most likely reason is weight decay: every configuration in the grid used weight decay 1e-4, while the default uses none, so the setting that works best was never part of the search. Within the grid, larger embeddings (64 → 32) and a learning rate of 0.01 were usually chosen, which agrees with the default being a good setting. If the default had been included as a seventh candidate, selection on validation would have kept it on every dataset, so the tuned rows should be read as a recorded unsuccessful experiment rather than as our best GAE/VGAE. A second search without weight decay is the obvious next step.

**Tuning changed the best configuration between settings.** On CiteSeer, the configuration with the best validation AUC (16 → 8, lr 0.001) is not the one with the best validation MRR (64 → 32, lr 0.01), so the two settings in that row come from different configurations. Easy and hard negatives reward different models even within one architecture.

**PubMed is hard for every model.** All models reach below 3 MRR in the HeaRT setting on PubMed, and GAE (2.82) is only slightly above Adamic–Adar (2.63). The larger graph gives each positive many plausible hard negatives.

**Ties hide the heuristics' AP.** Common Neighbors and Adamic–Adar give 0 to most pairs, so many positives tie with hundreds of hard negatives. This is why their HeaRT AP is close to 1% while their MRR is competitive; MRR handles ties by averaging the best and worst rank.

### Limitations and remaining difficulties

- **SEAL on PubMed did not finish.** Extracting and labelling a subgraph for every one of the 2.2 million hard negatives in the PubMed test set was too slow for our time budget. Only 2 of the 6 tuning configurations ran (validation only), so PubMed has no SEAL test result. The tuning rows are kept in `results/m2/raw_seal.csv`.
- **Small search spaces.** 6 configurations per model, one seed each, is a small budget. SEAL used only 1-hop subgraphs and 15 epochs; the original paper uses 2 hops.
- **One configuration for both settings (SEAL).** SEAL's configuration was chosen on validation HeaRT MRR and the same model is reported in the easy setting; GAE/VGAE were chosen separately for each setting.
- **Heuristics were not extended.** We planned to add Resource Allocation and Katz; they were not run in time.
- **One split.** All results are on HeaRT's single fixed split, so the standard deviations show seed-to-seed variation only.
- **Some hard negatives are true edges.** 0.1–0.4% of HeaRT's hard negatives are other held-out positives, because HeaRT only filters the edges known at that stage. We keep them as shipped so our numbers stay comparable with published HeaRT results.
- **Weight decay was fixed in the tuning grid.** All 6 GAE/VGAE configurations used weight decay 1e-4 and the untuned default was not a candidate, so the search could not find the best setting (see above).
- **Recorded failed attempt.** A first tuning attempt trained the models without scoring them, so it produced no usable results. It was replaced by `src/tune_m2.py`, which scores every configuration with the shared evaluator.

### How to reproduce Milestone 2

Each step is a Colab notebook (Runtime → Run all, GPU runtime) or a script run from the repository root. The HeaRT data is downloaded automatically on first use.

| Step | Notebook | Script | Output |
|---|---|---|---|
| Test the shared setup | `notebooks/m2_shared_setup_test.ipynb` | `python src/heart_data.py` | leak check |
| Baselines (default GAE, VGAE, CN, AA) | `notebooks/m2_baselines.ipynb` | `python src/baselines_m2.py` | `results/m2/raw_baselines.csv` |
| Tuning GAE / VGAE | `notebooks/m2_tuning.ipynb` | `python src/tune_m2.py` | `results/m2/raw_tuning.csv` |
| SEAL | `notebooks/m2_seal.ipynb` | | `results/m2/raw_seal.csv` |
| Final table | | `python src/aggregate_m2.py` | `results/m2/results_table.md`, `summary_results.csv` |

### Milestone 2 files

| File | Purpose |
|---|---|
| `src/heart_data.py` | Shared loader: HeaRT split, easy and hard negatives, leak check |
| `src/evaluate.py` | Shared evaluator: AUC, AP, MRR, Hits@K in both settings; results CSV writer |
| `src/baselines_m2.py` | Default GAE, VGAE, Common Neighbors and Adamic–Adar on the HeaRT split |
| `src/tune_m2.py` | Tuning of GAE and VGAE: 6 configurations, selection on validation, 10 seeds |
| `src/aggregate_m2.py` | Merges all result files and picks tuned configurations by validation score |
| `notebooks/m2_*.ipynb` | Colab notebooks: shared setup test, baselines, tuning (with outputs), SEAL |
| `results/m2/` | Raw runs, summary and final table |
| `docs/milestone2_handover.md` | Team task split and rules |

---

## Files in the repository

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── pubmed_and_aggregation.ipynb
├── src/
│   ├── autoencoder.py
│   └── baseline.py
├── results/
│   ├── raw_results.csv
│   └── summary_results.csv
├── report/
│   └── Experiment_Report.docx
└── docs/
    └── member2_citeseer_handover.txt
```

| File | What it does |
|---|---|
| `notebooks/pubmed_and_aggregation.ipynb` | **Main solution.** Installs PyTorch Geometric, creates 10 random edge splits per dataset, checks that no test edge leaks into training, trains GAE and VGAE, scores Common Neighbors and Adamic–Adar on the same test edges, then computes mean ± std and compares every result with the paper. |
| `src/autoencoder.py` | The official PyTorch Geometric GAE / VGAE example, unmodified. The notebook uses its model code; it can also be run on its own for a single run. |
| `src/baseline.py` | Member 1's original Common Neighbors and Adamic–Adar implementation (set-based). The notebook contains an equivalent vectorised version that runs on the same splits as the GNNs. |
| `results/raw_results.csv` | All 120 individual runs: dataset, model, run number, seed, AUC and AP. |
| `results/summary_results.csv` | Mean and standard deviation per dataset and model, with the paper's targets, the difference to them and whether each result is reproduced. |
| `report/Experiment_Report.docx` | The Milestone 1 group report with all tables and the discussion of the results. |
| `docs/member2_citeseer_handover.txt` | Member 2's handover notes from the first CiteSeer experiments. |
| `requirements.txt` | Python packages needed to run the code locally. |
| `.gitignore` | Keeps downloaded datasets, caches and checkpoints out of the repository. |

---

## How to reproduce

### Option A: Google Colab (recommended)

1. Open the notebook **[`notebooks/pubmed_and_aggregation.ipynb`](notebooks/pubmed_and_aggregation.ipynb)** in Colab (**File → Upload notebook**, or open it from GitHub).
2. Select a GPU: **Runtime → Change runtime type → T4 GPU**.
3. Run **Runtime → Run all**.

The notebook installs PyTorch Geometric, downloads the Planetoid datasets automatically, runs all 120 experiments (about 5–10 minutes on a T4), prints the final table in Markdown and LaTeX, and downloads `member3_raw_results.csv` and `summary_results.csv`.

To run a single dataset, edit the settings cell:

```python
DATASETS = ['Cora', 'CiteSeer', 'PubMed']   # e.g. ['PubMed'] for PubMed only
```

### Option B: run the official example directly

```bash
pip install -r requirements.txt
python src/autoencoder.py --dataset Cora                  # GAE
python src/autoencoder.py --dataset Cora --variational    # VGAE
```

The script prints AUC and AP after every epoch; the value at epoch 400 is the result. Each run draws a new random split and does not save it, so this mode **cannot** evaluate a baseline on the same hidden edges. That is why the notebook exists (see below).

### Requirements

- Python 3.10+
- PyTorch 2.x and PyTorch Geometric 2.4+ (pure `pip install torch_geometric`; no extra compiled packages are needed)
- scikit-learn, SciPy, NumPy, pandas

---

## Method

### Models

The notebook uses the model code of PyG's [`examples/autoencoder.py`](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/autoencoder.py) unchanged.

| Setting | Value |
|---|---|
| Encoder | 2-layer GCN, 32 hidden units, 16-dimensional embedding (VGAE: mean and log-std heads) |
| Decoder | Inner product |
| Loss | Binary cross-entropy on positive edges and sampled negative edges; VGAE adds a KL term weighted by 1/N |
| Optimizer | Adam, learning rate 0.01 |
| Epochs | 400 |
| Features | Row-normalised (`NormalizeFeatures`) |

### Data split

`RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True)` gives 85% training, 5% validation and 10% test edges, each test set paired with an equal number of random non-edges as negatives. Test and validation edges are **removed** from the graph used for message passing. The notebook checks this explicitly and asserts that zero test edges appear in the training graph.

### Evaluation protocol

- 10 runs per dataset with seeds 0–9, each giving a different random split.
- In every run, **GAE, VGAE, Common Neighbors and Adamic–Adar are all scored on the identical test positives and negatives**.
- Metrics: ROC-AUC and Average Precision at the final epoch, reported as mean ± sample standard deviation (`ddof=1`), as in the original paper.

### Heuristic baselines

Both heuristics are computed on the **training graph only**:

- **Common Neighbors:** CN(u, v) = |N(u) ∩ N(v)|
- **Adamic–Adar:** AA(u, v) = Σ over w ∈ N(u) ∩ N(v) of 1 / log |N(w)|

The notebook uses a vectorised sparse-matrix implementation, verified to give identical scores to NetworkX. `src/baseline.py` holds the original set-based implementation.

---

## Findings

**GAE matches the paper.** It lands within one standard deviation of the published values on every dataset, which confirms the pipeline and split are correct.

**VGAE falls short on Cora and CiteSeer.** Training length explains part of the gap: an earlier team run with 200 epochs gave only 83.27% AUC on Cora and 79.43% on CiteSeer, against 89.05% and 86.61% after 400. We attribute the rest to implementation differences between PyG and the original TensorFlow code. In particular, PyG trains with sampled negative edges instead of reconstructing the full weighted adjacency matrix. We have not yet tested this directly. In our runs GAE beat VGAE in all 30 paired runs, the opposite ordering to the paper on Cora and CiteSeer.

**The heuristics fail on these sparse graphs.** Random negative pairs almost never share a neighbour, so they all score 0, and many true test links have no common neighbour in the training graph either. The heuristic cannot rank those links, which is why Common Neighbors and Adamic–Adar give near-identical results, often with AUC equal to AP.

**Caveat.** These results use easy random negatives and untuned models, which is the evaluation setting HeaRT criticises. The GNN advantage is likely overstated here; Milestone 2 tests this.

---

## Roadmap

- [x] **Milestone 1 – Reproduce:** GAE / VGAE on Cora, CiteSeer and PubMed, 10 runs each; heuristic baselines on the same splits
- [x] **Milestone 2 – Improve:** run the SEAL example; evaluate all models on [HeaRT's](https://github.com/Juanhui28/HeaRT) hard negatives; tune every model, including the heuristics, with equal effort
- [ ] **Milestone 3 – Contribute:** how much results change with harder test examples; whether a heuristic + GNN combination beats both; which node pairs GNNs win or lose on

---

## Contributions

| Member | Role | Contribution |
|---|---|---|
| Member 1: Song Ma | Setup and heuristic baseline | Environment setup, `autoencoder.py` pipeline, Common Neighbors / Adamic–Adar implementation (`baseline.py`), first Cora runs |
| Member 2: İlknur Pehlivan | Small-dataset experiments | GAE and VGAE on CiteSeer (10 runs), handover notes |
| Member 3: Bhuwaneshwaree Ragavan | Large dataset and aggregation | PubMed experiments; final 10-run pipeline with all models on identical splits for all datasets; aggregation, results table and paper comparison |


**Milestone 2**

| Member | Contribution |
|---|---|
| İlknur Pehlivan | SEAL on the shared HeaRT split: adaptation of the PyG example, tuning (6 configurations) and 10-seed runs (`results/m2/raw_seal.csv`) |
| Song Ma | Tuning design for GAE and VGAE: the configuration grid (learning rate, layer sizes, epochs, weight decay) used in `src/tune_m2.py` |
| Bhuwaneshwaree Ragavan | Shared loader, evaluator and leak check (`src/heart_data.py`, `src/evaluate.py`); baselines on the HeaRT split; tuning pipeline and runs; merge script, results table and README |

---

## References

- T. N. Kipf, M. Welling. [Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308). NeurIPS Workshop on Bayesian Deep Learning, 2016.
- T. N. Kipf, M. Welling. [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907). ICLR, 2017.
- M. Zhang, Y. Chen. [Link Prediction Based on Graph Neural Networks](https://arxiv.org/abs/1802.09691) (SEAL). NeurIPS, 2018.
- J. Li et al. [Evaluating Graph Neural Networks for Link Prediction: Current Pitfalls and New Benchmarking](https://arxiv.org/abs/2306.10453) (HeaRT). NeurIPS, 2023. Code: [Juanhui28/HeaRT](https://github.com/Juanhui28/HeaRT)
- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric), `examples/autoencoder.py`

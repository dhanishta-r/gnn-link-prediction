# Do Graph Neural Networks Really Beat Counting Common Neighbours?

Group project for the Deep Learning course at BME (Budapest University of Technology and Economics).

**Link prediction** means guessing which missing connections exist in a network, as used in friend and product recommendation, citation suggestion and drug-interaction discovery. Many papers report that graph neural networks (GNNs) beat simple rules, but the [HeaRT study (NeurIPS 2023)](https://arxiv.org/abs/2306.10453) found problems in how these comparisons were made: baselines performing below their real level, inconsistent data splits, and test examples that were too easy.

Our question: **when do GNNs really win?**

This repository currently contains **Milestone 1: Reproduce**.

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

## Repository structure

```
.
├── README.md
├── requirements.txt
├── notebooks/
│   └── member3_pubmed_and_aggregation.ipynb   # main pipeline: all runs + aggregation (run in Colab)
├── src/
│   ├── autoencoder.py      # official PyG example (GAE / VGAE), unmodified
│   └── baseline.py         # Common Neighbors and Adamic-Adar (Member 1)
├── results/
│   ├── raw_results.csv     # 120 rows: dataset, model, run, seed, auc, ap
│   └── summary_results.csv # mean ± std per dataset and model, with paper targets
├── report/
│   └── Experiment_Report.docx   # Milestone 1 group report
└── docs/
    └── member2_citeseer_handover.txt
```

---

## How to reproduce

### Option A: Google Colab (recommended)

1. Open [`notebooks/member3_pubmed_and_aggregation.ipynb`](notebooks/member3_pubmed_and_aggregation.ipynb) in Colab (**File → Upload notebook**, or open it from GitHub).
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
- [ ] **Milestone 2 – Improve:** run the SEAL example; evaluate all models on [HeaRT's](https://github.com/Juanhui28/HeaRT) hard negatives; tune every model, including the heuristics, with equal effort
- [ ] **Milestone 3 – Contribute:** how much results change with harder test examples; whether a heuristic + GNN combination beats both; which node pairs GNNs win or lose on

---

## Team

| Member | Role | Contribution |
|---|---|---|
| Member 1: Song Ma - SAGUWI | Setup and heuristic baseline | Environment setup, `autoencoder.py` pipeline, Common Neighbors / Adamic–Adar implementation (`baseline.py`), first Cora runs |
| Member 2: İlknur Pehlivan - JE5XG1 | Small-dataset experiments | GAE and VGAE on CiteSeer (10 runs), handover notes |
| Member 3: Bhuwaneshwaree Ragavan - EWX1Q5 | Large dataset and aggregation | PubMed experiments; final 10-run pipeline with all models on identical splits for all datasets; aggregation, results table and paper comparison |

---

## References

- T. N. Kipf, M. Welling. [Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308). NeurIPS Workshop on Bayesian Deep Learning, 2016.
- T. N. Kipf, M. Welling. [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907). ICLR, 2017.
- M. Zhang, Y. Chen. [Link Prediction Based on Graph Neural Networks](https://arxiv.org/abs/1802.09691) (SEAL). NeurIPS, 2018.
- J. Li et al. [Evaluating Graph Neural Networks for Link Prediction: Current Pitfalls and New Benchmarking](https://arxiv.org/abs/2306.10453) (HeaRT). NeurIPS, 2023. Code: [Juanhui28/HeaRT](https://github.com/Juanhui28/HeaRT)
- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric), `examples/autoencoder.py`

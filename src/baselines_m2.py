"""Milestone 2 baselines on the HeaRT split ("before tuning" rows).

Runs the Milestone 1 models unchanged on the fixed HeaRT split and scores them
in both settings (easy and HeaRT hard negatives):

  CN, AA      Common Neighbours and Adamic-Adar (no training, one row each)
  GAE, VGAE   the official PyTorch Geometric autoencoder.py settings:
              GCN encoder 32 -> 16, Adam lr 0.01, 400 epochs, last epoch,
              row-normalised features

Command line
    python src/baselines_m2.py --datasets cora citeseer pubmed --seeds 10

The training function `train_autoencoder` takes the hyper-parameters as
arguments, so it can also be used for tuning (pass other values and a
different `config` string).
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from evaluate import append_results, evaluate_model
from heart_data import check_no_leak, load_heart

DEFAULT_OUT = os.path.join("results", "m2", "raw_baselines.csv")


# --------------------------------------------------------------------------
# heuristics
# --------------------------------------------------------------------------

def make_common_neighbours(data):
    A = data.adj

    def score(edges):
        return np.asarray(A[edges[:, 0]].multiply(A[edges[:, 1]]).sum(axis=1)).ravel()

    return score


def make_adamic_adar(data):
    A = data.adj
    deg = np.asarray(A.sum(axis=0)).ravel()
    w = np.zeros_like(deg)
    w[deg > 1] = 1.0 / np.log(deg[deg > 1])       # each common neighbour counts 1 / log(degree)
    A_w = A.multiply(w).tocsr()

    def score(edges):
        return np.asarray(A[edges[:, 0]].multiply(A_w[edges[:, 1]]).sum(axis=1)).ravel()

    return score


HEURISTICS = {"CN": make_common_neighbours, "AA": make_adamic_adar}


# --------------------------------------------------------------------------
# GAE / VGAE
# --------------------------------------------------------------------------

def train_autoencoder(data, variational: bool = False, seed: int = 0, epochs: int = 400,
                      lr: float = 0.01, out_channels: int = 16, hidden_channels: int | None = None,
                      dropout: float = 0.0, weight_decay: float = 0.0, normalize_features: bool = True,
                      device=None):
    """Train GAE or VGAE on the training edges only and return a score function.

    Defaults are the official autoencoder.py settings used in Milestone 1.
    """
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GAE, VGAE, GCNConv

    hidden = hidden_channels or 2 * out_channels

    class Encoder(torch.nn.Module):
        def __init__(self, in_channels):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden)
            self.conv_mu = GCNConv(hidden, out_channels)
            self.conv_logstd = GCNConv(hidden, out_channels) if variational else None

        def forward(self, x, edge_index):
            x = F.dropout(x, p=dropout, training=self.training)
            x = self.conv1(x, edge_index).relu()
            x = F.dropout(x, p=dropout, training=self.training)
            if variational:
                return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
            return self.conv_mu(x, edge_index)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, edge_index = data.to_torch()
    if normalize_features:                          # same as T.NormalizeFeatures()
        x = x / x.sum(dim=1, keepdim=True).clamp(min=1.0)
    x, edge_index = x.to(device), edge_index.to(device)

    model = (VGAE if variational else GAE)(Encoder(x.size(1))).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        z = model.encode(x, edge_index)
        loss = model.recon_loss(z, edge_index)      # training edges vs. sampled negatives
        if variational:
            loss = loss + (1 / data.num_nodes) * model.kl_loss()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        z = model.encode(x, edge_index).cpu()       # for VGAE this is the mean in eval mode

    def score(edges):
        e = torch.as_tensor(np.asarray(edges))
        return (z[e[:, 0]] * z[e[:, 1]]).sum(-1)    # inner-product decoder (before the sigmoid)

    return score


# --------------------------------------------------------------------------
# run everything
# --------------------------------------------------------------------------

def run_baselines(datasets=("cora", "citeseer", "pubmed"), seeds=range(10), out_csv: str = DEFAULT_OUT,
                  root: str = "data", epochs: int = 400, member: str = "C", verbose: bool = True):
    """Run all baselines and append the rows to `out_csv`."""
    for name in datasets:
        data = load_heart(name, root=root)
        check_no_leak(data, verbose=False)
        if verbose:
            print(data.summary())

        for model_name, make in HEURISTICS.items():
            score = make(data)
            for split in ("valid", "test"):
                rows = evaluate_model(score, data, split=split)
                append_results(out_csv, rows, dataset=name, model=model_name, config="default", seed=0,
                               member=member, notes="deterministic")
            if verbose:
                print(f"  {model_name:5s} test: easy AUC {rows[0]['AUC']:.2f} | heart MRR {rows[1]['MRR']:.2f}")

        for model_name, variational in (("GAE", False), ("VGAE", True)):
            for seed in seeds:
                start = time.time()
                score = train_autoencoder(data, variational=variational, seed=seed, epochs=epochs)
                for split in ("valid", "test"):
                    rows = evaluate_model(score, data, split=split)
                    append_results(out_csv, rows, dataset=name, model=model_name, config="default", seed=seed,
                                   member=member, notes=f"epochs={epochs}")
                if verbose:
                    print(f"  {model_name:5s} seed {seed} test: easy AUC {rows[0]['AUC']:.2f} | "
                          f"heart MRR {rows[1]['MRR']:.2f}  ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["cora", "citeseer", "pubmed"])
    parser.add_argument("--seeds", type=int, default=10, help="number of seeds (0 .. seeds-1)")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--root", default="data")
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()
    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} already exists. Delete or rename it first, so runs are not duplicated.")
    run_baselines(args.datasets, range(args.seeds), args.out, args.root, args.epochs)

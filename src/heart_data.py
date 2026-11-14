"""Shared data loader for Milestone 2 (HeaRT splits and negatives).

Every model in Milestone 2 (GAE, VGAE, SEAL, heuristics) must load its data
through `load_heart`, so that all of them train on the same graph and are
tested on the same positive and negative edges.

The data comes from the HeaRT benchmark (Li et al., NeurIPS 2023):
https://github.com/Juanhui28/HeaRT

For each dataset the HeaRT archive contains
    train_pos.txt, valid_pos.txt, test_pos.txt   fixed 85 / 5 / 10 edge split
    valid_neg.txt, test_neg.txt                  random ("easy") negatives
    heart_valid_samples.npy, heart_test_samples.npy
                                                 hard negatives, shape
                                                 (num_positives, K, 2)
    gnn_feature                                  node features

Usage
-----
    from heart_data import load_heart, check_no_leak
    data = load_heart("cora")        # downloads the archive on first use
    check_no_leak(data)
    data.train_pos                   # (E, 2) numpy array, one row per edge
    data.adj                         # symmetric scipy CSR matrix (train only)
    data.easy_neg["test"]            # (N, 2)
    data.hard_neg["test"]            # (P, K, 2), row i belongs to test_pos[i]
    x, edge_index = data.to_torch()  # for PyTorch Geometric models

Rule: models may only see `train_pos` (as graph structure and as training
labels). `valid_*` is for model selection and tuning. `test_*` is used once,
for the final numbers.
"""

from __future__ import annotations

import glob
import os
import subprocess
import tarfile
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

HEART_URL = "https://zenodo.org/records/22184581/files/HeaRT.tar.gz"
DATASETS = ("cora", "citeseer", "pubmed")
PLANETOID_NAMES = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}


@dataclass
class LinkData:
    name: str
    num_nodes: int
    train_pos: np.ndarray            # (E_train, 2)
    valid_pos: np.ndarray            # (E_valid, 2)
    test_pos: np.ndarray             # (E_test, 2)
    easy_neg: dict                   # {"valid": (N, 2), "test": (N, 2)}
    hard_neg: dict                   # {"valid": (P, K, 2), "test": (P, K, 2)}
    adj: sp.csr_matrix               # symmetric, train edges only, no self loops
    folder: str
    easy_neg_source: str = "heart archive"
    _x: object = field(default=None, repr=False)

    def pos(self, split: str) -> np.ndarray:
        return {"train": self.train_pos, "valid": self.valid_pos, "test": self.test_pos}[split]

    def features(self):
        """Node feature matrix as a torch tensor (imports torch lazily)."""
        if self._x is None:
            self._x = _load_features(self.folder, self.name, self.num_nodes)
        return self._x

    def to_torch(self):
        """Return (x, edge_index) for PyTorch Geometric.

        edge_index holds the training edges in both directions.
        """
        import torch

        e = torch.from_numpy(self.train_pos.T.copy()).long()
        edge_index = torch.cat([e, e.flip(0)], dim=1)
        return self.features(), edge_index

    def summary(self) -> str:
        k = {s: self.hard_neg[s].shape[1] for s in ("valid", "test")}
        return (
            f"{self.name}: {self.num_nodes} nodes | "
            f"train {len(self.train_pos)} / valid {len(self.valid_pos)} / test {len(self.test_pos)} edges | "
            f"easy negatives valid {len(self.easy_neg['valid'])}, test {len(self.easy_neg['test'])} "
            f"({self.easy_neg_source}) | "
            f"hard negatives per positive: valid {k['valid']}, test {k['test']}"
        )


# --------------------------------------------------------------------------
# download
# --------------------------------------------------------------------------

def find_dataset_dir(root: str, name: str):
    """Find the folder that holds `<name>/train_pos.txt` anywhere under root."""
    hits = glob.glob(os.path.join(root, "**", name, "train_pos.txt"), recursive=True)
    return os.path.dirname(sorted(hits, key=len)[0]) if hits else None


def download_heart(root: str = "data") -> None:
    """Download and unpack the HeaRT archive into `root` (skipped if present)."""
    os.makedirs(root, exist_ok=True)
    if all(find_dataset_dir(root, n) for n in DATASETS):
        return
    archive = os.path.join(root, "HeaRT.tar.gz")
    if not os.path.exists(archive):
        print(f"Downloading {HEART_URL} ...")
        try:
            subprocess.run(["wget", "-q", "-O", archive, HEART_URL], check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            import urllib.request

            urllib.request.urlretrieve(HEART_URL, archive)
    print("Unpacking ...")
    with tarfile.open(archive) as tar:
        try:
            tar.extractall(root, filter="data")
        except TypeError:          # Python < 3.12
            tar.extractall(root)
    missing = [n for n in DATASETS if not find_dataset_dir(root, n)]
    if missing:
        raise RuntimeError(f"Archive unpacked but these datasets were not found: {missing}")


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def _read_edges(path: str) -> np.ndarray:
    edges = np.loadtxt(path, dtype=np.int64, delimiter="\t", ndmin=2)
    return edges[edges[:, 0] != edges[:, 1]]      # drop self loops, as HeaRT does


def _load_features(folder: str, name: str, num_nodes=None):
    import torch

    path = os.path.join(folder, "gnn_feature")
    if os.path.exists(path):
        try:
            obj = torch.load(path, map_location="cpu")
        except Exception:
            obj = torch.load(path, map_location="cpu", weights_only=False)
        x = obj["entity_embedding"] if isinstance(obj, dict) else obj
        x = torch.as_tensor(x).float()
    else:   # fall back to the Planetoid features (same node numbering)
        from torch_geometric.datasets import Planetoid

        x = Planetoid(os.path.join(folder, "_planetoid"), PLANETOID_NAMES[name])[0].x
    if num_nodes is not None and x.size(0) != num_nodes:
        raise RuntimeError(f"Feature matrix has {x.size(0)} rows but the graph has {num_nodes} nodes")
    return x


def _sample_easy_negatives(num: int, num_nodes: int, forbidden: set, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < num:
        s, t = rng.integers(0, num_nodes, size=2)
        if s == t or (min(s, t), max(s, t)) in forbidden:
            continue
        out.append((s, t))
    return np.asarray(out, dtype=np.int64)


def _num_nodes(folder: str, name: str, max_id: int):
    """Number of nodes = rows of the feature matrix (isolated nodes included).

    Without torch installed (heuristics only) it falls back to the largest
    node id in the edge lists + 1.
    """
    try:
        x = _load_features(folder, name)
        return int(x.size(0)), x
    except ImportError:
        return max_id + 1, None


def load_heart(name: str, root: str = "data", download: bool = True,
               hard_file: str = "samples.npy") -> LinkData:
    """Load one dataset with the fixed HeaRT split and both kinds of negatives."""
    name = name.lower()
    if name not in DATASETS:
        raise ValueError(f"name must be one of {DATASETS}")
    if download:
        download_heart(root)
    folder = find_dataset_dir(root, name)
    if folder is None:
        raise FileNotFoundError(f"No HeaRT data for '{name}' under '{root}'. Run download_heart('{root}').")

    pos = {s: _read_edges(os.path.join(folder, f"{s}_pos.txt")) for s in ("train", "valid", "test")}
    hard = {s: np.load(os.path.join(folder, f"heart_{s}_{hard_file}")).astype(np.int64)
            for s in ("valid", "test")}
    for s in ("valid", "test"):
        if hard[s].ndim != 3 or hard[s].shape[2] != 2:
            raise RuntimeError(f"Unexpected hard-negative shape {hard[s].shape} for {s}")
        if hard[s].shape[0] != len(pos[s]):
            raise RuntimeError(
                f"{s}: {hard[s].shape[0]} hard-negative rows but {len(pos[s])} positive edges")

    max_id = int(max(p.max() for p in pos.values()))
    num_nodes, x = _num_nodes(folder, name, max_id)
    if max_id >= num_nodes:
        raise RuntimeError("Edge list refers to a node id outside the feature matrix")

    all_pos = set()
    for p in pos.values():
        all_pos.update(zip(np.minimum(p[:, 0], p[:, 1]).tolist(), np.maximum(p[:, 0], p[:, 1]).tolist()))

    easy, source = {}, "heart archive"
    for i, s in enumerate(("valid", "test")):
        path = os.path.join(folder, f"{s}_neg.txt")
        if os.path.exists(path):
            easy[s] = _read_edges(path)
        else:
            easy[s] = _sample_easy_negatives(len(pos[s]), num_nodes, all_pos, seed=2026 + i)
            source = "sampled here, seed 2026"

    tr = pos["train"]
    rows = np.concatenate([tr[:, 0], tr[:, 1]])
    cols = np.concatenate([tr[:, 1], tr[:, 0]])
    adj = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(num_nodes, num_nodes))
    adj.data[:] = 1.0            # duplicate edges count once

    return LinkData(name=name, num_nodes=num_nodes, train_pos=pos["train"], valid_pos=pos["valid"],
                    test_pos=pos["test"], easy_neg=easy, hard_neg=hard, adj=adj, folder=folder,
                    easy_neg_source=source, _x=x)


# --------------------------------------------------------------------------
# leak check
# --------------------------------------------------------------------------

def _pair_keys(edges: np.ndarray, num_nodes: int) -> np.ndarray:
    e = edges.reshape(-1, 2)
    return np.minimum(e[:, 0], e[:, 1]) * num_nodes + np.maximum(e[:, 0], e[:, 1])


def check_no_leak(data: LinkData, verbose: bool = True) -> dict:
    """Check that nothing used for validation or testing is visible in training.

    Raises AssertionError if a validation or test positive is in the training
    graph, or if the validation and test positives overlap. Also reports how
    many negatives are in fact training edges (HeaRT allows none on these
    datasets, so this should be 0).
    """
    n = data.num_nodes
    train = np.unique(_pair_keys(data.train_pos, n))
    valid = np.unique(_pair_keys(data.valid_pos, n))
    test = np.unique(_pair_keys(data.test_pos, n))
    report = {
        "valid_pos_in_train": int(np.isin(valid, train).sum()),
        "test_pos_in_train": int(np.isin(test, train).sum()),
        "valid_pos_in_test": int(np.isin(valid, test).sum()),
    }
    all_pos = np.concatenate([train, valid, test])
    for s in ("valid", "test"):
        report[f"easy_{s}_neg_that_are_real_edges"] = int(np.isin(_pair_keys(data.easy_neg[s], n), all_pos).sum())
        report[f"hard_{s}_neg_that_are_real_edges"] = int(np.isin(_pair_keys(data.hard_neg[s], n), all_pos).sum())
    if verbose:
        print(f"Leak check for {data.name}:")
        for k, v in report.items():
            print(f"  {k:36s} {v}")
    assert report["valid_pos_in_train"] == 0, "validation edges are in the training graph"
    assert report["test_pos_in_train"] == 0, "test edges are in the training graph"
    assert report["valid_pos_in_test"] == 0, "validation and test edges overlap"
    return report


if __name__ == "__main__":
    for ds in DATASETS:
        d = load_heart(ds)
        print(d.summary())
        check_no_leak(d)

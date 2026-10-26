import torch
from sklearn.metrics import roc_auc_score, average_precision_score

def build_neighbor_sets(train_pos_edge_index, num_nodes):
    """
    Build neighbor sets using ONLY training edges. DO NOT access test edges!
    train_pos_edge_index: PyG training positive edge index
    return: list of neighbor sets for each node
    """
    neighbors = [set() for _ in range(num_nodes)]
    row, col = train_pos_edge_index
    for u, v in zip(row.tolist(), col.tolist()):
        neighbors[u].add(v)
    return neighbors


def score_edges(neighbors, edge_index, method="cn"):
    """
    Score a batch of candidate edges.
    method="cn" : Common Neighbors
    method="aa" : Adamic-Adar
    """
    row, col = edge_index
    scores = []
    for u, v in zip(row.tolist(), col.tolist()):
        common = neighbors[u] & neighbors[v]
        if method == "cn":
            score = float(len(common))
        elif method == "aa":
            score = 0.0
            for w in common:
                deg_w = len(neighbors[w])
                if deg_w > 1:
                    score += 1.0 / torch.log(torch.tensor(deg_w, dtype=torch.float)).item()
        else:
            raise NotImplementedError("Only cn or aa supported")
        scores.append(score)
    return torch.tensor(scores)


def evaluate_baseline(data, method="cn"):
    """
    Input: complete data object processed by train_test_split_edges (the same data object used for GAE!)
    Output: auc, ap
    """
    num_nodes = data.num_nodes

    neighbors = build_neighbor_sets(data.train_pos_edge_index, num_nodes)

    pos_scores = score_edges(neighbors, data.test_pos_edge_index, method=method)
    neg_scores = score_edges(neighbors, data.test_neg_edge_index, method=method)

    y_true = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)]).numpy()
    y_score = torch.cat([pos_scores, neg_scores]).numpy()

    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    return auc, ap

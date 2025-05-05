import torch
import networkx as nx
import numpy as np

def label_propagation(graph, y, mask, alpha=0.99, max_iter=1000, tol=1e-4):
    n = graph.number_of_nodes()
    labels = torch.unique(y[mask])
    c = len(labels)

    y_onehot = torch.zeros(n, c)
    for i in range(n):
        if mask[i]:
            y_onehot[i, (labels == y[i]).nonzero(as_tuple=True)[0]] = 1.0

    A = nx.to_numpy_array(graph)
    D_inv = np.diag(1.0 / (A.sum(axis=1) + 1e-8))  # avoid division by zero
    W = D_inv @ A
    W = torch.from_numpy(W).float()

    f = y_onehot.clone()

    for _ in range(max_iter):
        f_new = alpha * W @ f + (1 - alpha) * y_onehot
        if torch.norm(f_new - f) < tol:
            break
        f = f_new

    return f.argmax(dim=1)

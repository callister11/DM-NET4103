import networkx as nx
import torch
import numpy as np
from label_propagation import label_propagation
from evaluate import evaluate

def load_graph_gml(path):
    G = nx.read_gml(path, label='id')
    attributes = {
        'gender': [],
        'major': [],
        'dorm': []
    }

    for node in G.nodes(data=True):
        attributes['gender'].append(int(node[1].get('gender', -1)))
        attributes['major'].append(int(node[1].get('major', -1)))
        attributes['dorm'].append(int(node[1].get('dorm', -1)))

    return G, attributes

def run_experiment(attr_name, attr_values, G, fractions=[0.1, 0.2, 0.3]):
    print(f"\n==== {attr_name.upper()} ====")
    results = []
    attr_values = torch.tensor(attr_values, dtype=torch.long)
    valid_mask = attr_values >= 0  # Filter out invalid values
    if valid_mask.sum() == 0:
        print(f"No valid data for attribute '{attr_name}'. Skipping...")
        return results

    attr_values = attr_values[valid_mask]
    subgraph = G.subgraph(np.nonzero(valid_mask.numpy())[0]).copy()
    if len(subgraph.nodes) == 0:
        print(f"Subgraph for attribute '{attr_name}' is empty. Skipping...")
        return results

    for frac in fractions:
        n = len(attr_values)
        mask = torch.rand(n) > frac
        if mask.sum() == 0:
            print(f"All nodes are masked for fraction {frac}. Skipping...")
            continue

        pred = label_propagation(subgraph, attr_values, mask)
        acc, mae = evaluate(attr_values, pred, mask)
        print(f"Missing {int(frac*100)}% → Accuracy: {acc:.3f} | MAE: {mae:.3f}")
        results.append((frac, acc, mae))
    return results

if __name__ == "__main__":
    G, attributes = load_graph_gml("../data/MIT8.gml")
    for attr in ['gender', 'major', 'dorm']:
        run_experiment(attr, attributes[attr], G)

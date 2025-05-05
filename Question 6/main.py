import networkx as nx
from networkx.algorithms.community import label_propagation_communities
from sklearn.metrics import normalized_mutual_info_score

def load_graph_and_labels(path, attribute):
    G = nx.read_gml(path, label='id')
    labels = []
    valid_nodes = []

    for node_id, attr in G.nodes(data=True):
        value = attr.get(attribute)
        if value is not None:
            labels.append(int(value))
            valid_nodes.append(node_id)

    G = G.subgraph(valid_nodes).copy()
    return G, labels

def detect_communities(G):
    communities = list(label_propagation_communities(G))
    label_map = {}
    for i, community in enumerate(communities):
        for node in community:
            label_map[node] = i
    return [label_map[n] for n in G.nodes()]

def run(path, attributes=['dorm', 'gender', 'major']):
    for attr in attributes:
        print(f"\n=== Testing community detection vs {attr.upper()} ===")
        G, labels = load_graph_and_labels(path, attr)
        predicted = detect_communities(G)
        nmi = normalized_mutual_info_score(labels, predicted)
        print(f"NMI (Communities vs {attr}): {nmi:.3f}")

if __name__ == "__main__":
    run("../data/MIT8.gml")

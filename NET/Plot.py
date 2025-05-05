import networkx as nx
import matplotlib.pyplot as plt

def load_lcc_graph(filepath):
    G = nx.read_gml(filepath)
    if not nx.is_connected(G):
        # Take the Largest Connected Component (LCC)
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    return G

def plot_degree_distribution(G, title):
    plt.figure(figsize=(10, 6))
    degrees = [d for _, d in G.degree()]
    plt.hist(degrees, bins=50, density=True, alpha=0.6)
    plt.xlabel("Degree")
    plt.ylabel("Probability Density")
    plt.title(f"Degree Distribution of {title} Network")
    plt.grid(True)
    plt.show()

def compute_network_metrics(G):
    # Global Clustering Coefficient
    global_clustering_coefficient = nx.transitivity(G)

    # Mean Local Clustering Coefficient
    mean_local_clustering_coefficient = nx.average_clustering(G)

    # Edge Density
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    edge_density = num_edges / (num_nodes * (num_nodes - 1) / 2)

    return global_clustering_coefficient, mean_local_clustering_coefficient, edge_density

# Load graphs
caltech = load_lcc_graph("fb100/data/Caltech36.gml")
mit = load_lcc_graph("fb100/data/MIT8.gml")
jhu = load_lcc_graph("fb100/data/JohnsHopkins55.gml")

# Plot individual distributions
plot_degree_distribution(caltech, "Caltech")
plot_degree_distribution(mit, "MIT")
plot_degree_distribution(jhu, "Johns Hopkins")

# Compute and print metrics for each graph
for network, name in zip([caltech, mit, jhu], ['Caltech', 'MIT', 'Johns Hopkins']):
    global_clustering, mean_local_clustering, edge_density = compute_network_metrics(network)
    print(f"{name} Network Metrics:")
    print(f"  Global Clustering Coefficient: {global_clustering:.4f}")
    print(f"  Mean Local Clustering Coefficient: {mean_local_clustering:.4f}")
    print(f"  Edge Density: {edge_density:.4f}")
    
    # Determine if the network is sparse based on edge density
    if edge_density < 0.01:
        print(f"  {name} network is sparse.\n")
    else:
        print(f"  {name} network is not sparse.\n")

print("Metrics computed successfully.")

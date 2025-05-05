import networkx as nx
import matplotlib.pyplot as plt

def load_lcc_graph(filepath):
    G = nx.read_gml(filepath)
    if not nx.is_connected(G):
        # Take the Largest Connected Component (LCC)
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    return G

def plot_degree_vs_clustering(G, title):
    plt.figure(figsize=(10, 6))
    degrees = [d for _, d in G.degree()]
    clustering_coeffs = list(nx.clustering(G).values())
    
    plt.scatter(degrees, clustering_coeffs, alpha=0.6, edgecolors='w', s=50)
    plt.xlabel("Degree")
    plt.ylabel("Local Clustering Coefficient")
    plt.title(f"Degree vs Local Clustering Coefficient of {title} Network")
    plt.grid(True)
    plt.show()

# Load graphs
caltech = load_lcc_graph("fb100/data/Caltech36.gml")
mit = load_lcc_graph("fb100/data/MIT8.gml")
jhu = load_lcc_graph("fb100/data/JohnsHopkins55.gml")

# Plot Degree vs Local Clustering Coefficient
plot_degree_vs_clustering(caltech, "Caltech")
plot_degree_vs_clustering(mit, "MIT")
plot_degree_vs_clustering(jhu, "Johns Hopkins")

print("Degree vs Clustering Coefficient scatter plots generated.")

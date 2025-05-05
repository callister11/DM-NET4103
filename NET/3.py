import networkx as nx
import matplotlib.pyplot as plt
import os
import numpy as np

def load_graph_from_gml(filepath):
    G = nx.read_gml(filepath)
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    return G

def compute_assortativity(G, attribute):
    # Attempt to access the first node and check if the attribute exists
    first_node = next(iter(G.nodes(data=True)))  # Safely get the first node and its data
    if attribute not in first_node[1]:  # Check if the attribute exists in node data
        return None  # If the attribute doesn't exist
    return nx.attribute_assortativity_coefficient(G, attribute)

def plot_assortativity_vs_size(assortativity_values, sizes, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(sizes, assortativity_values, alpha=0.7)
    plt.xscale('log')
    plt.yscale('linear')
    plt.xlabel("Network Size (n)")
    plt.ylabel("Assortativity")
    plt.title(title)
    plt.axhline(0, color='red', linestyle='--', label="No assortativity")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_assortativity_distribution(assortativity_values, title):
    plt.figure(figsize=(10, 6))
    plt.hist(assortativity_values, bins=20, alpha=0.7, density=True)
    plt.axvline(0, color='red', linestyle='--', label="No assortativity")
    plt.xlabel("Assortativity")
    plt.ylabel("Density")
    plt.title(f"Assortativity Distribution - {title}")
    plt.legend()
    plt.grid(True)
    plt.show()

def analyze_networks(folder_path):
    assortativity_results = {
        'size': [],
        'student_fac': [],
        'gender': [],
        'major_index': [],
        'dorm': [],
    }

    for filename in os.listdir(folder_path):
        if filename.endswith(".gml"):
            filepath = os.path.join(folder_path, filename)
            G = load_graph_from_gml(filepath)
            n = len(G.nodes())
            assortativity_results['size'].append(n)

            # Assortativity by different attributes
            for attribute in ['student_fac', 'gender', 'major_index', 'dorm']:
                assortativity = compute_assortativity(G, attribute)
                if assortativity is not None:
                    assortativity_results[attribute].append(assortativity)

    return assortativity_results

# Folder path to your .gml files
folder_path = 'fb100/data/'

# Analyze networks
assortativity_results = analyze_networks(folder_path)

# Plot the assortativity vs. network size for each attribute
for attribute in ['student_fac', 'gender', 'major_index', 'dorm']:
    plot_assortativity_vs_size(assortativity_results[attribute], assortativity_results['size'], f'{attribute.capitalize()} Assortativity vs. Network Size')

# Plot the distribution of assortativity for each attribute
for attribute in ['student_fac', 'gender', 'major_index', 'dorm']:
    plot_assortativity_distribution(assortativity_results[attribute], f'{attribute.capitalize()} Assortativity Distribution')

import networkx as nx
import matplotlib.pyplot as plt
import os
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_graph_from_gml(filepath):
    logging.info(f"Loading graph from {filepath}")
    G = nx.read_gml(filepath)
    if not nx.is_connected(G):
        logging.warning(f"Graph in {filepath} is not connected. Extracting the largest connected component.")
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    return G

def compute_assortativity(G, attribute):
    logging.info(f"Computing assortativity for attribute '{attribute}'")
    first_node = next(iter(G.nodes(data=True)))  # Safely get the first node and its data
    if attribute not in first_node[1]:  # Check if the attribute exists in node data
        logging.warning(f"Attribute '{attribute}' not found in graph nodes. Skipping.")
        return None  # If the attribute doesn't exist
    return nx.attribute_assortativity_coefficient(G, attribute)

def plot_assortativity_vs_size(assortativity_values, sizes, title):
    logging.info(f"Plotting assortativity vs. size for {title}")
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
    logging.info(f"Plotting assortativity distribution for {title}")
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
    logging.info(f"Analyzing networks in folder: {folder_path}")
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
            logging.info(f"Processing file: {filename}")
            G = load_graph_from_gml(filepath)
            n = len(G.nodes())
            logging.info(f"Graph size (number of nodes): {n}")
            assortativity_results['size'].append(n)

            # Assortativity by different attributes
            for attribute in ['student_fac', 'gender', 'major_index', 'dorm']:
                assortativity = compute_assortativity(G, attribute)
                if assortativity is not None:
                    assortativity_results[attribute].append(assortativity)

    return assortativity_results

# Folder path to your .gml files
folder_path = 'data/'

# Analyze networks
logging.info("Starting network analysis")
assortativity_results = analyze_networks(folder_path)

# Plot the assortativity vs. network size for each attribute
for attribute in ['student_fac', 'gender', 'major_index', 'dorm']:
    plot_assortativity_vs_size(assortativity_results[attribute], assortativity_results['size'], f'{attribute.capitalize()} Assortativity vs. Network Size')

# Plot the distribution of assortativity for each attribute
for attribute in ['student_fac', 'gender', 'major_index', 'dorm']:
    plot_assortativity_distribution(assortativity_results[attribute], f'{attribute.capitalize()} Assortativity Distribution')

logging.info("Network analysis and plotting completed")
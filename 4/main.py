import os
import networkx as nx
import random
import logging
from tqdm import tqdm
from metrics import CommonNeighbors, Jaccard, AdamicAdar
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_graphs_from_folder(folder_path):
    logging.info(f"Loading all graphs from folder: {folder_path}")
    graphs = []
    file_names = []
    files = [file for file in os.listdir(folder_path) if file.endswith(".gml")]  # No limit on files
    for file in files:
        path = os.path.join(folder_path, file)
        try:
            logging.info(f"Loading graph from file: {file}")
            G = nx.read_gml(path)
            G = nx.convert_node_labels_to_integers(G)
            graphs.append(G)
            file_names.append(file)
        except Exception as e:
            logging.warning(f"Failed to load {file}: {e}")

    logging.info(f"Loaded {len(graphs)} graphs.")
    return graphs, file_names

def remove_random_edges(G, fraction):
    logging.info(f"Removing {fraction * 100:.1f}% of edges from the graph.")
    G_copy = G.copy()
    edges = list(G_copy.edges())
    num_remove = int(len(edges) * fraction)
    removed_edges = random.sample(edges, num_remove)
    G_copy.remove_edges_from(removed_edges)
    logging.info(f"Removed {num_remove} edges.")
    return G_copy, removed_edges

def evaluate_prediction(predicted_edges, removed_edges, k_list):
    logging.info("Evaluating predictions.")
    results = []
    removed_set = set(tuple(sorted(e)) for e in removed_edges)
    predicted_sorted = sorted(predicted_edges, key=lambda x: x[1], reverse=True)

    for k in k_list:
        logging.info(f"Evaluating top-{k} predictions.")
        top_k_edges = set(tuple(sorted(pair)) for pair, score in predicted_sorted[:k])
        tp = len(removed_set & top_k_edges)
        fp = k - tp
        fn = len(removed_set) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        results.append({
            "k": k,
            "top@k": tp,
            "precision@k": precision,
            "recall@k": recall
        })

        logging.info(f"Results for k={k}: top@k={tp}, precision={precision:.2f}, recall={recall:.2f}")

    return results

def run_experiments(folder_path, fractions=[0.05, 0.1], k_list=[50, 100, 200, 400]):
    logging.info(f"Running experiments on folder: {folder_path}")
    classes = [CommonNeighbors, Jaccard, AdamicAdar]

    # Load all graphs
    graphs, names = load_graphs_from_folder(folder_path)

    for G, name in zip(graphs, names):
        logging.info(f"Processing graph: {name}")
        for frac in fractions:
            G_train, removed = remove_random_edges(G, frac)
            logging.info(f"Removed {len(removed)} edges (fraction = {frac}).")

            for cls in classes:
                logging.info(f"Running model: {cls.__name__}")
                model = cls(G_train)
                predictions = model.fit()
                scores = evaluate_prediction(predictions, removed, k_list)

                logging.info(f"Results for {cls.__name__}:")
                for result in scores:
                    logging.info(f"  k={result['k']} | top@k={result['top@k']} | precision={result['precision@k']:.2f} | recall={result['recall@k']:.2f}")

if __name__ == "__main__":
    logging.info("Starting the script.")
    run_experiments("../data/")
    logging.info("Script completed.")

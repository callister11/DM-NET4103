from abc import ABC, abstractmethod
import networkx as nx
import numpy as np
import math  # Ensure math is imported
import random

class LinkPrediction(ABC):
    def __init__(self, graph):
        """
        Parameters
        ----------
        graph : NetworkX graph
        """
        self.graph = graph
        self.N = len(graph)

    def neighbors(self, v):
        return list(self.graph.neighbors(v))

    @abstractmethod
    def fit(self):
        raise NotImplementedError("Fit must be implemented")
    
class CommonNeighbors(LinkPrediction):
    def __init__(self, graph):
        super().__init__(graph)

    def fit(self):
        scores = []
        nodes = list(self.graph.nodes())

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                u, v = nodes[i], nodes[j]
                if self.graph.has_edge(u, v):
                    continue
                neighbors_u = set(self.neighbors(u))
                neighbors_v = set(self.neighbors(v))
                common = neighbors_u & neighbors_v
                score = len(common)
                scores.append(((u, v), score))

        return scores

class Jaccard(LinkPrediction):
    def __init__(self, graph):
        super().__init__(graph)

    def fit(self):
        scores = []
        nodes = list(self.graph.nodes())

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                u, v = nodes[i], nodes[j]
                if self.graph.has_edge(u, v):
                    continue
                neighbors_u = set(self.neighbors(u))
                neighbors_v = set(self.neighbors(v))
                union_size = len(neighbors_u | neighbors_v)
                if union_size == 0:
                    score = 0.0
                else:
                    score = len(neighbors_u & neighbors_v) / union_size
                scores.append(((u, v), score))

        return scores

class AdamicAdar(LinkPrediction):
    def __init__(self, graph):
        super().__init__(graph)

    def fit(self):
        scores = []
        nodes = list(self.graph.nodes())

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                u, v = nodes[i], nodes[j]
                if self.graph.has_edge(u, v):
                    continue
                neighbors_u = set(self.neighbors(u))
                neighbors_v = set(self.neighbors(v))
                common_neighbors = neighbors_u & neighbors_v

                score = 0.0
                for w in common_neighbors:
                    degree = len(self.neighbors(w))
                    if degree > 1:
                        score += 1 / math.log(degree)  # Uses math.log

                scores.append(((u, v), score))

        return scores
import sys

import pickle
import networkx as nx
import os
import pandas as pd
import pandas as pd
import numpy as np
from qubosolver import QUBOInstance
from qubosolver.solver import QuboSolver
from qubosolver.config import SolverConfig, ClassicalConfig

results_path = " "

def load_graphs_from_folder(folder_path):
    graphs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".graphml"):
            file_path = os.path.join(folder_path, filename)
            G = nx.read_graphml(file_path)
            graphs.append(G)
    return graphs

def is_independent_set(graph, subset) -> bool:
        """
        Checks if the given subset of nodes is an independent set in the graph.

        :param graph: The input graph.
        :param subset: The list of nodes to check.
        :return: True if the subset is an independent set, False otherwise.
        """
        return not any(
                
            graph.has_edge(subset[i], subset[j])
            for i in range(len(subset))
            for j in range(i + 1, len(subset))
        )

def calculate_weight(graph, node_list ) -> float:
        """
        Calculates the total weight of a given list of nodes in the graph.

        :param node_list: List of nodes.
        :return: Total weight of the nodes.
        """
        total_weight: float = 0
        for node in node_list:
            if "weight" in graph.nodes[node]:
                total_weight += graph.nodes[node]["weight"]
            else:
                raise ValueError(f"Node {node} does not have a 'weight' attribute.")
        return total_weight

def to_qubo(graph, penalty: float | None = None) -> np.ndarray:
        """Convert a MISInstance to a qubo matrix.

        QUBO formulation:
        Minimize:
            Q(x) = -∑_{i ∈ V} w_i x_i  +  λ ∑_{(i, j) ∈ E} x_i x_j

        Args:
            penalty (float, optional): Penalty factor. Defaults to None.

        Raises:
            ValueError: When penalty is strictly inferior to 2 x max(weight).

        Returns:
            np.ndarray: The QUBO matrix formulation of MIS.
        """

        # Linear terms: -sum_i w_i x_i
        
        weights = [float(graph.nodes[n].get("weight", 1)) for n in graph.nodes]
        max_Q = max(weights)
        if penalty is None:
            penalty = 2.5 * max_Q
        elif penalty < 2.0 * max_Q:
            raise ValueError("Penalty must be greater than 2 x max(weight).")

        # Quadratic terms: penalty sum_ij x_i x_j
        Q = nx.adjacency_matrix(graph, weight=None).toarray() * penalty
        Q -= np.diag(np.array(weights))

        return Q



folder_path = " "
graph_list = load_graphs_from_folder(folder_path)


graphs_experiments = []
for graph in graph_list:
    if  graph.number_of_nodes() < 447:
         graphs_experiments.append(graph)

data = []

for G in graphs_experiments:

    G_complement = nx.complement(G)
    G_complement.add_nodes_from(G.nodes(data=True))

    qubo_matrix = to_qubo(G_complement)

    # Keep track of the node names in order
    node_mapping = list(G_complement.nodes)

    n_nodes = G_complement.number_of_nodes()

    raw_weights = [data.get('weight', 1.0) for _, data in G_complement.nodes(data=True)]
    max_w = max(raw_weights)
    min_w = min(raw_weights)

        # B. Heuristic SA Parameters
    # 1. Initial Temp: High enough to allow flips (~ 2x max weight)
    h_initial_temp = max_w * 2.0
    
    # 2. Final Temp: Low enough to freeze (~ 0.1x min weight)
    h_final_temp = max(1e-4, min_w * 0.1)
    
    # 3. Max Iterations: Scale with graph size
    h_max_iter = n_nodes * 1000 
    
    print(f"Nodes: {n_nodes} | T_init: {h_initial_temp:.4f} | T_final: {h_final_temp:.4f} | Iters: {h_max_iter}")

    qubo = QUBOInstance(coefficients=qubo_matrix)

    config = SolverConfig(use_quantum = False, classical=
                          ClassicalConfig(classical_solver_type="simulated_annealing", 
                                          max_bitstrings=10,
                                          max_iter = h_max_iter,
                                          sa_initial_temp = h_initial_temp,
                                          sa_final_temp= h_final_temp
                                          ))
    
    solver = QuboSolver(qubo, config)
    opt_value = solver.solve()

    solutions = []
    for sol in opt_value.bitstrings:
         solution_values = opt_value.bitstrings.flatten().tolist()
         selected_nodes = [name for name, val in zip(node_mapping, solution_values) if val == 1]
         solutions.append(selected_nodes)

    for set in solutions:
         if is_independent_set(G_complement, set):
              final_sol = set
              sa_value = calculate_weight(G_complement, final_sol)
              break
    
                
    dict = {
            "sa_value": sa_value,
            "number_nodes": G_complement.number_of_nodes(),
            "number_edges": G_complement.number_of_edges(),
            "solution_sa": final_sol, 
            }
                
    data.append(dict)
    df = pd.DataFrame(data)
    df.to_csv(f"{results_path}/EXPERIMENT_.csv", index=False)
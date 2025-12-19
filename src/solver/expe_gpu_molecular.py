import sys

from greedy_subgraph_vv import greedy_subgraph_solver_vv
from classic_MIS_solvers import solve_weighted_mis, weighted_greedy_independent_set, weighted_generate_different_mis
from quantum_solver_molecular import q_solver
import pickle
import networkx as nx
import os
import pandas as pd


results_path = "..."

def load_graphs_from_folder(folder_path):
    graphs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".graphml"):
            file_path = os.path.join(folder_path, filename)
            G = nx.read_graphml(file_path)
            graphs.append(G)
    return graphs

# Exemple d'utilisation
folder_path = "/Molecular-Docking/data/instances/complement"
graph_list = load_graphs_from_folder(folder_path)



with open('../coordinate_arrays.pickle', 'rb') as handle:
    fresnel_id_coords_dic = pickle.load(handle)

rydberg_blockade = 6.6

G = nx.Graph()

for graph in graph_list:
    if graph.number_of_nodes() == 540:
        G = graph

cplex_sol = solve_weighted_mis(G)
greedy_sol = weighted_greedy_independent_set(G)

data = []
for nb_sg in [1,2,3,4,5]:
    for nb_mis in [1,2,3,4,5]:
        for exploration in [10, 15]:

            print("nb_sg", nb_sg)
            print("nb_mis", nb_mis)


            solver = greedy_subgraph_solver_vv(G, fresnel_id_coords_dic, rydberg_blockade, q_solver)

            Not_branched = solver.solve(method='breadthfirst',exact_solving_threshold = 5,
                                        subgraph_quantity = nb_sg, mis_sample_quantity = nb_mis, 
                                        exploration_threshold=exploration)

            quantum_value =  solver.calculate_weight(Not_branched[0])
            greedy_value =  solver.calculate_weight(greedy_sol)
            opt_value =  solver.calculate_weight(cplex_sol)

            graph_solved_list_edges = Not_branched[1]
            graph_solved_list_nodes = Not_branched[2]
            

            dict = {
                "quantum_value": quantum_value,
                "greedy_value": greedy_value,
                "opt_value": opt_value,
                "size": G.number_of_nodes(),
                "nb_sg": nb_sg,
                "nb_mis": nb_mis,
                "expl_thres:": exploration,
                "graph_solved_list_num_edges": graph_solved_list_edges,
                "graph_solved_list_num_nodes": graph_solved_list_nodes,
                "solution_quantum": Not_branched[0], 
                "solution_greedy": greedy_sol,
                "solution_cplex": cplex_sol
            }
            data.append(dict)
            df = pd.DataFrame(data)
            df.to_csv(f"{results_path}/EXPERIMENT_solution_TACE_AS_COMPL_size_{G.number_of_nodes()}.csv", index=False)
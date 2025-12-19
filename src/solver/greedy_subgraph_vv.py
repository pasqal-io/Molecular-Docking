import sys

sys.path.append("/home/vonofre/Documents/greedy_subgraph_mis/")

import copy
import math
from typing import Any, Callable, Dict, List, Set, Tuple, Union
from heapq import heappush, heappop


from collections import deque
from typing import Tuple

import networkx as nx
from networkx import Graph
from greedy_lattice_mapping import GreedyMapping, Lattice
from classic_MIS_solvers import solve_weighted_mis, weighted_greedy_independent_set, weighted_generate_different_mis


class greedy_subgraph_solver_vv:
    def __init__(
        self,
        graph: Graph,
        lattice_id_coord_dic: Dict[int, Tuple[float, float]],
        rydberg_blockade: float,
        mis_solving_function: Callable[[Graph, int], List[Set[int]]],
        seed: int = 0,
    ) -> None:
        """
        Initializes the greedy_subgraph_solver with the input graph, lattice parameters,
        and a function for solving the maximum independent set (MIS) problem.

        :param graph: The input graph.
        :param lattice_id_coord_dic: Dictionary mapping lattice IDs to coordinates.
        :param rydberg_blockade: The Rydberg blockade radius.
        :param mis_solving_function: Function to solve the MIS problem on a given graph.
        :param seed: Seed for randomness.
        """
        self.lattice: Lattice = Lattice(
            lattice_id_coord_dic,
            rydberg_blockade,
            seed=seed,
        )
        self.lattice_id_coord_dic: Dict[int, Tuple[float, float]] = lattice_id_coord_dic
        self.graph: Graph = graph
        self.mis_solving_function: Callable[[Graph, int], List[Set[int]]] = mis_solving_function

    def obtain_embeddable_subgraphs(
        self,
        current_graph: Graph,
        subgraph_quantity: int,
    ) -> List[Dict[int, int]]:
        """
        Generates a list of embeddable subgraphs based on greedy mapping, sorted by size.

        :param current_graph: The input graph from which subgraphs are extracted.
        :param subgraph_quantity: Number of largest subgraphs to return.
        :return: List of mappings representing subgraphs.
        """
        mappings: List[Dict[int, int]] = []
        for node in current_graph.nodes():
            greedy_mapper: GreedyMapping = GreedyMapping(
                current_graph,
                copy.deepcopy(self.lattice),
                {},
            )
            subgraph_mapping: Dict[int, int] = greedy_mapper.generate_greedy_ud_subgraph_with(node)
            mappings.append(subgraph_mapping)

        return sorted(mappings, key=lambda x: len(x), reverse=True)[:subgraph_quantity]

    def remove_mis_open_neighborhood_nodes_from_graph(
        self,
        current_graph: Graph,
        mis: List[int],
    ) -> Graph:
        """
        Removes MIS nodes and their open neighborhood from the graph.

        :param current_graph: The input graph.
        :param mis: List of nodes in the MIS.
        :return: A new graph with MIS nodes and their neighbors removed.
        """
        new_subgraph_with_removed_nodes: Graph = copy.deepcopy(current_graph)
        nodes_to_remove: Set[int] = set(mis)
        for node in mis:
            current_node_neighbors: Set[int] = set(current_graph.neighbors(node))
            nodes_to_remove.update(current_node_neighbors)

        new_subgraph_with_removed_nodes.remove_nodes_from(nodes_to_remove)
        return new_subgraph_with_removed_nodes

    def calculate_weight(self, node_list: List[int]) -> float:
        """
        Calculates the total weight of a given list of nodes in the graph.

        :param node_list: List of nodes.
        :return: Total weight of the nodes.
        """
        total_weight: float = 0
        for node in node_list:
            if "weight" in self.graph.nodes[node]:
                total_weight += self.graph.nodes[node]["weight"]
            else:
                raise ValueError(f"Node {node} does not have a 'weight' attribute.")
        return total_weight

    def is_independent_set(self, graph: Graph, subset: List[int]) -> bool:
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

    def backtracking_independent_set(
        self,
        graph: Graph,
        subset: List[int],
        index: int,
        max_set: List[List[int]],
    ) -> None:
        """
        Uses backtracking to find the maximum weighted independent set.

        :param graph: The input graph.
        :param subset: The current subset of nodes being explored.
        :param index: The current index in the graph nodes.
        :param max_set: A list containing the current maximum independent set (by reference).
        """
        nodes: List[int] = list(graph.nodes())
        current_weight: float = self.calculate_weight(subset)
        max_weight: float = self.calculate_weight(max_set[0])

        if self.is_independent_set(graph, subset) and current_weight > max_weight:
            max_set[0] = subset[:]

        for i in range(index, len(nodes)):
            subset.append(nodes[i])
            self.backtracking_independent_set(graph, subset, i + 1, max_set)
            subset.pop()

    def find_maximum_independent_set(self, graph: Graph) -> List[int]:
        """
        Finds the maximum weighted independent set in the graph using backtracking.

        :param graph: The input graph with weighted nodes.
        :return: A list of nodes representing the maximum weighted independent set.
        """
        max_set: List[List[int]] = [[]]
        self.backtracking_independent_set(graph, [], 0, max_set)
        return max_set[0]

    def generate_graph_to_solve(
        self,
        current_graph: Graph,
        lattice: Graph,
        mapping: Dict[int, int],
    ) -> Graph:
        """
        Generates a new subgraph to solve based on a mapping between the current graph
        and the lattice.

        :param current_graph: The input graph.
        :param lattice: The lattice graph.
        :param mapping: Mapping of nodes from the input graph to the lattice.
        :return: The generated subgraph.
        """
        subgraph: Graph = nx.Graph()
        for source_node, target_node in mapping.items():
            current_weight: Union[float, None] = current_graph.nodes[source_node].get("weight")
            lattice_pos: Union[Tuple[float, float], None] = lattice.nodes[target_node].get("pos")
            subgraph.add_node(
                target_node,
                weight=current_weight,
                pos=lattice_pos,
            )
        for source_node, target_node in mapping.items():
            for neighbor in lattice.neighbors(target_node):
                if neighbor in mapping.values():
                    subgraph.add_edge(target_node, neighbor)

        return subgraph

    def solve_depthfirst(
        self,
        current_graph: Graph,
        exact_solving_threshold: int,
        subgraph_quantity: int,
        mis_sample_quantity: int,
    ) -> List[int]:
        """
        Recursively solves the graph for the maximum independent set using a hybrid
        approach combining greedy subgraph extraction and exact solving.

        :param current_graph: The graph to solve.
        :param exact_solving_threshold: Size threshold for exact solving.
        :param subgraph_quantity: Number of subgraphs to consider.
        :param mis_sample_quantity: Number of MIS samples to compute.
        :return: The maximum weighted independent set.
        """
        if len(current_graph) <= exact_solving_threshold:
            return self.find_maximum_independent_set(current_graph)

        mappings: List[Dict[int, int]] = self.obtain_embeddable_subgraphs(
            current_graph,
            subgraph_quantity,
        )
        current_max_mis: List[int] = []

        for mapping in mappings:
            nx_subgraph: Graph = current_graph.subgraph(mapping.keys())
            if nx_subgraph.number_of_nodes() <= exact_solving_threshold:
                self.solve_depthfirst(
                    nx_subgraph,
                    exact_solving_threshold,
                    subgraph_quantity,
                    mis_sample_quantity,
                )
                continue

            graph_to_solve: Graph = self.generate_graph_to_solve(
                current_graph,
                self.lattice.lattice,
                mapping,
            )
            current_mis_set_on_lattice: List[Set[int]] = self.mis_solving_function(
                graph_to_solve,
                self.lattice_id_coord_dic,
                mis_sample_quantity,
            )
            inverse_mapping: Dict[int, int] = {v: k for k, v in mapping.items()}
            current_mis_set: List[List[int]] = [
                [inverse_mapping[value] for value in mis_lattice]
                for mis_lattice in current_mis_set_on_lattice
            ]
            for current_mis in current_mis_set:
                new_subgraph_with_removed_nodes: Graph = (
                    self.remove_mis_open_neighborhood_nodes_from_graph(
                        current_graph,
                        current_mis,
                    )
                )

                mis_from_recursive_call: List[int] = self.solve_depthfirst(
                    new_subgraph_with_removed_nodes,
                    exact_solving_threshold,
                    subgraph_quantity,
                    mis_sample_quantity,
                )

                if (
                    self.calculate_weight(current_mis)
                    + self.calculate_weight(mis_from_recursive_call)
                    > self.calculate_weight(current_max_mis)
                ):
                    current_max_mis = current_mis + mis_from_recursive_call

        if not self.is_independent_set(current_graph, current_max_mis):
            raise Exception("Not an independent set!")

        return current_max_mis


    def solve_breadthfirst(
        self,
        initial_graph: Graph,
        exact_solving_threshold: int,
        subgraph_quantity: int,
        mis_sample_quantity: int,
        exploration_threshold = int,
        graph_solved_data_edges = List,
        graph_solved_data_nodes = List,
    ) -> List[int]:
        """
        Breadth-first solution for Maximum Independent Set using hybrid approximation.

        :param initial_graph: The initial graph.
        :param exact_solving_threshold: Threshold for exact solving.
        :param subgraph_quantity: Number of subgraphs per graph state.
        :param mis_sample_quantity: Number of MIS samples per subgraph.
        :param exploration_threshold: Max number of subgraphs to consider before pruning.
        :return: The best found independent set.
        """
        
        # Queue of tuples: (graph_state, current_MIS_list)
        queue = deque([(initial_graph, [])])
        best_mis: List[int] = []
        exploration_counter = 0
        print("Initial graph size", len(initial_graph))
        while queue:
            #print(queue)
            next_level = []

            while queue:
                current_graph, current_mis = queue.popleft()

                if len(current_graph) <= exact_solving_threshold:
                    exact_mis = self.find_maximum_independent_set(current_graph)
                    total_mis = current_mis + exact_mis
                    #print("testing")
                    if self.calculate_weight(total_mis) > self.calculate_weight(best_mis):
                        #print("Testing best mis")
                        best_mis = total_mis
                    continue

                mappings = self.obtain_embeddable_subgraphs(
                    current_graph,
                    subgraph_quantity,
                )

                for mapping in mappings:
                    subgraph = current_graph.subgraph(mapping.keys())
                    

                    if subgraph.number_of_nodes() <= exact_solving_threshold:
                        print("start solving classical",len(subgraph))
                        solved = self.find_maximum_independent_set(subgraph)
                        new_graph = self.remove_mis_open_neighborhood_nodes_from_graph(
                            current_graph, solved
                        )
                        next_level.append((new_graph, current_mis + solved))
                        continue

                    graph_to_solve = self.generate_graph_to_solve(
                        current_graph,
                        self.lattice.lattice,
                        mapping,
                    )
                    graph_solved_data_edges.append(graph_to_solve.number_of_edges())
                    graph_solved_data_nodes.append(graph_to_solve.number_of_nodes())

                    print("start solving quantum",graph_to_solve.number_of_nodes())

                    mis_candidates_on_lattice = self.mis_solving_function(
                        graph_to_solve,
                        self.lattice_id_coord_dic,
                        mis_sample_quantity,
                    )

                    inverse_mapping = {v: k for k, v in mapping.items()}
                    mis_candidates = [
                        [inverse_mapping[node] for node in mis] for mis in mis_candidates_on_lattice
                    ]

                    for mis_set in mis_candidates:
                        new_graph = self.remove_mis_open_neighborhood_nodes_from_graph(
                            current_graph, mis_set
                        )
                        next_level.append((new_graph, current_mis + mis_set))
                        exploration_counter += 1
                        #print("exploration counter", exploration_counter)

            # Prune or keep best states
            if exploration_counter > exploration_threshold:
                next_level.sort(key=lambda x: self.calculate_weight(x[1]), reverse=True)
                next_level = next_level[:subgraph_quantity]
                exploration_counter = 0 

            queue = deque(next_level)

        print("reconstructed",self.is_independent_set(initial_graph, best_mis))

        return best_mis, graph_solved_data_edges, graph_solved_data_nodes
    



    def solve_breadthfirst_2_sol(
        self,
        initial_graph: Graph,
        exact_solving_threshold: int,
        subgraph_quantity: int,
        mis_sample_quantity: int,
        exploration_threshold: int,
        graph_solved_data_edges: List,
        graph_solved_data_nodes: List,
    ) -> Tuple[List[int], List[int], List, List]:
        """
        Breadth-first solution that finds two best solutions by creating and running
        two independent branches from the start.

        :param initial_graph: The initial graph.
        :param exact_solving_threshold: Threshold for exact solving.
        :param subgraph_quantity: Number of subgraphs per graph state.
        :param mis_sample_quantity: Number of MIS samples per subgraph.
        :param exploration_threshold: Max number of subgraphs to consider before pruning.
        :return: A tuple containing the best MIS from branch 1, the best MIS from branch 2,
                    and lists of graph edge/node data.
        """
        # --- STEP 1: GENERATE INITIAL STATES TO CREATE THE TWO BRANCHES ---
        print("Initial graph size", len(initial_graph))
        
        # Run the processing logic once to get the first set of candidate states
        next_level_candidates = []
        temp_queue = deque([(initial_graph, [])])
        
        current_graph, current_mis = temp_queue.popleft()
        
        mappings = self.obtain_embeddable_subgraphs(
            current_graph,
            subgraph_quantity,
        )

        for mapping in mappings:
            subgraph = current_graph.subgraph(mapping.keys())
            
            if subgraph.number_of_nodes() <= exact_solving_threshold:
                solved = self.find_maximum_independent_set(subgraph)
                new_graph = self.remove_mis_open_neighborhood_nodes_from_graph(current_graph, solved)
                next_level_candidates.append((new_graph, current_mis + solved))
                continue
            
            graph_to_solve = self.generate_graph_to_solve(
                current_graph, self.lattice.lattice, mapping,
            )
            # The following lists are shared and collect data from all branches
            graph_solved_data_edges.append(graph_to_solve.number_of_edges())
            graph_solved_data_nodes.append(graph_to_solve.number_of_nodes())
            
            mis_candidates_on_lattice = self.mis_solving_function(
                graph_to_solve, self.lattice_id_coord_dic, mis_sample_quantity,
            )

            inverse_mapping = {v: k for k, v in mapping.items()}
            mis_candidates = [
                [inverse_mapping[node] for node in mis] for mis in mis_candidates_on_lattice
            ]

            for mis_set in mis_candidates:
                new_graph = self.remove_mis_open_neighborhood_nodes_from_graph(current_graph, mis_set)
                next_level_candidates.append((new_graph, current_mis + mis_set))

        # Sort candidates to find the best two starting points
        next_level_candidates.sort(key=lambda x: self.calculate_weight(x[1]), reverse=True)

        # Initialize the two branches
        queue1, queue2 = deque(), deque()
        best_mis_branch1, best_mis_branch2 = [], []

        if len(next_level_candidates) > 0:
            start_state1 = next_level_candidates[0]
            queue1.append(start_state1)
            best_mis_branch1 = start_state1[1]
            print(f"Branch 1 starts with MIS weight: {self.calculate_weight(best_mis_branch1)}")

        if len(next_level_candidates) > 1:
            start_state2 = next_level_candidates[1]
            queue2.append(start_state2)
            best_mis_branch2 = start_state2[1]
            print(f"Branch 2 starts with MIS weight: {self.calculate_weight(best_mis_branch2)}")
        
        # --- STEP 2: RUN THE BREADTH-FIRST SEARCH ON BOTH BRANCHES ---
        while queue1 or queue2:
            
            # --- Process one level of Branch 1 ---
            if queue1:
                next_level1 = []
                exploration_counter1 = 0
                while queue1:
                    current_graph_1, current_mis_1 = queue1.popleft()

                    if len(current_graph_1) <= exact_solving_threshold:
                        exact_mis_1 = self.find_maximum_independent_set(current_graph_1)
                        total_mis_1 = current_mis_1 + exact_mis_1
                        if self.calculate_weight(total_mis_1) > self.calculate_weight(best_mis_branch1):
                            best_mis_branch1 = total_mis_1
                        continue

                    mappings_1 = self.obtain_embeddable_subgraphs(current_graph_1, subgraph_quantity)
                    for mapping_1 in mappings_1:
                        # ... (Full processing logic for branch 1)
                        # This is a direct copy of the original logic, using branch-1 variables
                        subgraph_1 = current_graph_1.subgraph(mapping_1.keys())
                        if subgraph_1.number_of_nodes() <= exact_solving_threshold:
                            solved_1 = self.find_maximum_independent_set(subgraph_1)
                            new_graph_1 = self.remove_mis_open_neighborhood_nodes_from_graph(current_graph_1, solved_1)
                            next_level1.append((new_graph_1, current_mis_1 + solved_1))
                            continue
                        
                        graph_to_solve_1 = self.generate_graph_to_solve(current_graph_1, self.lattice.lattice, mapping_1)
                        graph_solved_data_edges.append(graph_to_solve_1.number_of_edges())
                        graph_solved_data_nodes.append(graph_to_solve_1.number_of_nodes())
                        
                        mis_cand_lattice_1 = self.mis_solving_function(graph_to_solve_1, self.lattice_id_coord_dic, mis_sample_quantity)
                        inv_map_1 = {v: k for k, v in mapping_1.items()}
                        mis_cand_1 = [[inv_map_1[node] for node in mis] for mis in mis_cand_lattice_1]
                        
                        for mis_set_1 in mis_cand_1:
                            new_graph_1 = self.remove_mis_open_neighborhood_nodes_from_graph(current_graph_1, mis_set_1)
                            next_level1.append((new_graph_1, current_mis_1 + mis_set_1))
                            exploration_counter1 += 1
                
                # Prune and update queue1
                if exploration_counter1 > exploration_threshold:
                    next_level1.sort(key=lambda x: self.calculate_weight(x[1]), reverse=True)
                    next_level1 = next_level1[:subgraph_quantity]
                queue1 = deque(next_level1)

            # --- Process one level of Branch 2 ---
            if queue2:
                next_level2 = []
                exploration_counter2 = 0
                while queue2:
                    current_graph_2, current_mis_2 = queue2.popleft()

                    if len(current_graph_2) <= exact_solving_threshold:
                        exact_mis_2 = self.find_maximum_independent_set(current_graph_2)
                        total_mis_2 = current_mis_2 + exact_mis_2
                        if self.calculate_weight(total_mis_2) > self.calculate_weight(best_mis_branch2):
                            best_mis_branch2 = total_mis_2
                        continue

                    mappings_2 = self.obtain_embeddable_subgraphs(current_graph_2, subgraph_quantity)
                    for mapping_2 in mappings_2:
                        # ... (Full processing logic for branch 2)
                        subgraph_2 = current_graph_2.subgraph(mapping_2.keys())
                        if subgraph_2.number_of_nodes() <= exact_solving_threshold:
                            solved_2 = self.find_maximum_independent_set(subgraph_2)
                            new_graph_2 = self.remove_mis_open_neighborhood_nodes_from_graph(current_graph_2, solved_2)
                            next_level2.append((new_graph_2, current_mis_2 + solved_2))
                            continue
                        
                        graph_to_solve_2 = self.generate_graph_to_solve(current_graph_2, self.lattice.lattice, mapping_2)
                        graph_solved_data_edges.append(graph_to_solve_2.number_of_edges())
                        graph_solved_data_nodes.append(graph_to_solve_2.number_of_nodes())
                        
                        mis_cand_lattice_2 = self.mis_solving_function(graph_to_solve_2, self.lattice_id_coord_dic, mis_sample_quantity)
                        inv_map_2 = {v: k for k, v in mapping_2.items()}
                        mis_cand_2 = [[inv_map_2[node] for node in mis] for mis in mis_cand_lattice_2]
                        
                        for mis_set_2 in mis_cand_2:
                            new_graph_2 = self.remove_mis_open_neighborhood_nodes_from_graph(current_graph_2, mis_set_2)
                            next_level2.append((new_graph_2, current_mis_2 + mis_set_2))
                            exploration_counter2 += 1

                # Prune and update queue2
                if exploration_counter2 > exploration_threshold:
                    next_level2.sort(key=lambda x: self.calculate_weight(x[1]), reverse=True)
                    next_level2 = next_level2[:subgraph_quantity]
                queue2 = deque(next_level2)

        # --- STEP 3: RETURN FINAL RESULTS FROM BOTH BRANCHES ---
        print("Branch 1 best MIS reconstructed:", self.is_independent_set(initial_graph, best_mis_branch1))
        print("Branch 2 best MIS reconstructed:", self.is_independent_set(initial_graph, best_mis_branch2))

        return best_mis_branch1, best_mis_branch2, graph_solved_data_edges, graph_solved_data_nodes


    def solve(
        self,
        method: str,
        exact_solving_threshold: int = 10,
        subgraph_quantity: int = 5,
        mis_sample_quantity: int = 1,
        exploration_threshold: int =5,
    ) -> List[int]:
        """
        Solves the maximum independent set problem on the input graph.

        :param exact_solving_threshold: Size threshold for exact solving.
        :param subgraph_quantity: Number of subgraphs to consider.
        :param mis_sample_quantity: Number of MIS samples to compute.
        :return: The maximum weighted independent set.
        """
        if method=='depthfirst':
            graph_solved_data = []
            return self.solve_depthfirst(
                self.graph,
                exact_solving_threshold,
                subgraph_quantity,
                mis_sample_quantity)
        if method== 'breadthfirst_2_sol':
            graph_solved_data_edges = []
            graph_solved_data_nodes = []
            return self.solve_breadthfirst_2_sol(
                self.graph,
                exact_solving_threshold,
                subgraph_quantity,
                mis_sample_quantity,
                exploration_threshold,
                graph_solved_data_edges,
                graph_solved_data_nodes
            )
        elif method=='breadthfirst':
            graph_solved_data_edges = []
            graph_solved_data_nodes = []
            return self.solve_breadthfirst(
                self.graph,
                exact_solving_threshold,
                subgraph_quantity,
                mis_sample_quantity,
                exploration_threshold,
                graph_solved_data_edges,
                graph_solved_data_nodes
            )
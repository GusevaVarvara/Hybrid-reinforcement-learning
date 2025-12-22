import os
import random
import math
from concurrent.futures import ThreadPoolExecutor
import networkx as nx


def initialize_pheromones(graph, k):
    """
    Initializes pheromones for all edges in the graph.

    Args:
        graph (any): The graph object representing the network, which must have an 'edges' attribute and edge weights.
        k (float): A constant used to initialize the pheromone levels.

    Returns:
        Dict[Tuple[int, int], float]: A dictionary with edges as keys and initial pheromone levels as values.
    """
    pheromones = {}
    for u, v in graph.edges:
        edge = (min(u, v), max(u, v))
        distance = graph[u][v]["weight"]
        pheromones[edge] = k / distance
    return pheromones


def update_pheromones(
    pheromones,
    paths,
    graph,
    iteration,
    total_iterations,
    min_rate,
    max_rate,
    max_pheromone=100,
):
    """
    Updates the pheromone levels based on evaporation, path density and contributions.

    Args:
        pheromones (Dict[Tuple[int, int], float]): Current pheromone levels for each edge.
        paths (List[List[int]]): List of paths taken by ants during the current iteration.
        graph (any): The graph representing the environment in which ants are moving.
        iteration (int): The current iteration number.
        total_iterations (int): The total number of iterations.
        min_rate (float): The minimum rate of evaporation.
        max_rate (float): The maximum rate of evaporation.
        max_pheromone (float, optional): The maximum pheromone level for an edge (default is 100).

    Returns:
        Dict[Tuple[int, int], float]: The updated pheromone levels for all edges.
    """
    evaporation_rate = min_rate + (max_rate - min_rate) * math.exp(
        -iteration / total_iterations
    )

    for edge in pheromones:
        pheromones[edge] *= 1 - evaporation_rate
        pheromones[edge] = max(pheromones[edge], 1e-5)

    path_lengths = []
    edge_usage = {edge: 0 for edge in pheromones}

    for path in paths:
        length = sum(
            graph[path[i]][path[i + 1]]["weight"] for i in range(len(path) - 1)
        )
        path_lengths.append(length)
        for i in range(len(path) - 1):
            edge = (min(path[i], path[i + 1]), max(path[i], path[i + 1]))
            edge_usage[edge] += 1

    avg_length = sum(path_lengths) / len(path_lengths) if path_lengths else 1

    for path in paths:
        path_length = sum(
            graph[path[i]][path[i + 1]]["weight"] for i in range(len(path) - 1)
        )
        contribution = math.log(1 + avg_length / path_length) * 0.125
        for i in range(len(path) - 1):
            edge = (min(path[i], path[i + 1]), max(path[i], path[i + 1]))
            density_factor = edge_usage[edge] / len(paths)
            pheromones[edge] += contribution / (1 + density_factor)
            pheromones[edge] = min(pheromones[edge], max_pheromone)

    return pheromones


def adjust_parameters(
    iteration, max_iterations, alpha_start, beta_start, alpha_end, beta_end
):
    """
    Smoothly adjusts the alpha and beta parameters over iterations.

    Args:
        iteration (int): The current iteration number.
        max_iterations (int): The total number of iterations.
        alpha_start (float): The initial value of alpha.
        beta_start (float): The initial value of beta.
        alpha_end (float): The final value of alpha.
        beta_end (float): The final value of beta.

    Returns:
        Tuple[float, float]: The adjusted values of alpha and beta.
    """
    alpha = alpha_start + (alpha_end - alpha_start) * (iteration / max_iterations)
    beta = beta_start + (beta_end - beta_start) * (iteration / max_iterations)
    return alpha, beta


def simulate_ant(start_node, end_node, graph, pheromones, alpha, beta):
    """
     Simulates an ant's movement from the start node to the end node.

     Args:
         start_node (int): The starting node for the ant.
         end_node (int): The destination node for the ant.
         graph (any): The graph in which the ant moves.
         pheromones (Dict[Tuple[int, int], float]): The pheromone levels for each edge.
         alpha (float): The influence of pheromone on decision-making.
         beta (float): The influence of distance on decision-making.

     Returns:
         Optional[List[int]]: The path taken by the ant, or None if no valid path is found.
     """
    path = [start_node]
    current_node = start_node
    visited_nodes = {start_node}

    while current_node != end_node:
        next_node = move_ant(
            current_node, visited_nodes, graph, pheromones, alpha, beta, end_node
        )
        if next_node is None:
            break
        path.append(next_node)
        visited_nodes.add(next_node)
        current_node = next_node

    return path if path[-1] == end_node else None


def move_ant(current_node, visited_nodes, graph, pheromones, alpha, beta, end_node):
    """
    Chooses the next node for the ant to visit based on pheromone levels and distance.

    Args:
        current_node (int): The current node the ant is at.
        visited_nodes (set): A set of nodes that the ant has already visited.
        graph (any): The graph in which the ant is moving.
        pheromones (Dict[Tuple[int, int], float]): The pheromone levels for each edge.
        alpha (float): The influence of pheromone on decision-making.
        beta (float): The influence of distance on decision-making.

    Returns:
        Optional[int]: The next node for the ant to visit, or None if no valid node is available.
    """
    unvisited_nodes = set(graph.neighbors(current_node)) - visited_nodes
    if not unvisited_nodes:
        return None

    if random.random() < 0.02:
        return random.choice(list(unvisited_nodes))

    probabilities = calculate_transition_probabilities(
        current_node, unvisited_nodes, graph, pheromones, alpha, beta, end_node
    )
    nodes, probs = zip(*probabilities)
    return random.choices(nodes, weights=probs, k=1)[0]


def calculate_transition_probabilities(
        current_node, unvisited_nodes, graph, pheromones, alpha, beta, end_node
):
    probabilities = []
    total = 0

    end_pos = graph.nodes[end_node]['pos']

    for node in unvisited_nodes:
        edge = (min(current_node, node), max(current_node, node))
        pheromone = pheromones.get(edge, 0)

        node_pos = graph.nodes[node]['pos']
        dist_to_goal = abs(node_pos[0] - end_pos[0]) + abs(node_pos[1] - end_pos[1])
        heuristic = 1.0 / (dist_to_goal + 0.1)

        prob_weight = (pheromone ** alpha) * (heuristic ** beta)
        total += prob_weight

    for node in unvisited_nodes:
        edge = (min(current_node, node), max(current_node, node))
        pheromone = pheromones.get(edge, 0)

        node_pos = graph.nodes[node]['pos']
        dist_to_goal = abs(node_pos[0] - end_pos[0]) + abs(node_pos[1] - end_pos[1])
        heuristic = 1.0 / (dist_to_goal + 0.1)

        probability = ((pheromone ** alpha) * (heuristic ** beta)) / total
        probabilities.append((node, probability))

    return probabilities


def update_graph(graph, pheromones, changes):
    """
    Updates the graph and pheromone levels based on changes.

    Args:
        graph (any): The graph object to be updated.
        pheromones (Dict[Tuple[int, int], float]): The pheromone levels for each edge.
        changes (List[Tuple[int, int, Optional[float]]]): A list of changes to be applied to the graph, where each change is a tuple (u, v, new_weight).
    """
    for u, v, new_weight in changes:
        edge = (min(u, v), max(u, v))
        if new_weight is None:
            if graph.has_edge(u, v):
                graph.remove_edge(u, v)
            pheromones.pop(edge, None)
        else:
            if graph.has_edge(u, v):
                old_weight = graph[u][v]["weight"]
                graph[u][v]["weight"] = new_weight
                if edge in pheromones:

                    pheromones[edge] *= old_weight / new_weight
            else:
                graph.add_edge(u, v, weight=new_weight)
                pheromones[edge] = 1 / new_weight


def is_path_valid(path, graph):
    """
    Checks if all edges in the given path are valid (i.e., they exist in the current graph).

    Args:
        path (List[int]): The list of nodes representing the path.
        graph (any): The graph in which the path is being checked.

    Returns:
        bool: True if all edges in the path are valid, False otherwise.
    """
    for i in range(len(path) - 1):
        if not graph.has_edge(path[i], path[i + 1]):
            return False
    return True


def run_ant_colony_dynamic(
    graph,
    num_ants_start=None,
    num_ants_end=None,
    num_iterations=50,
    start_node=None,
    end_node=None,
    dynamic_changes=None,
    num_threads=None,
    min_rate=0.05, # 0.05
    max_rate=0.2, # 0.25
    alpha_start=0.5, # 0.8
    beta_start=3.0, # 4.0
    alpha_end=2.0, # 2.0
    beta_end=5.0, # 2.0
):
    """
    Runs the ant colony optimization algorithm for dynamic graphs.

    Args:
        graph (any): The graph representing the network.
        num_ants_start (Optional[int]): The starting number of ants.
        num_ants_end (Optional[int]): The ending number of ants.
        num_iterations (int): The number of iterations to run the algorithm.
        start_node (Optional[int]): The starting node for the ant(s).
        end_node (Optional[int]): The ending node for the ant(s).
        dynamic_changes (Optional[Dict[int, List[Tuple[int, int, Optional[float]]]]]): A dictionary of dynamic changes to the graph.
        num_threads (Optional[int]): The number of threads to use for parallel execution.
        min_rate (float): The minimum rate of evaporation for pheromones.
        max_rate (float): The maximum rate of evaporation for pheromones.
        alpha_start (float): The starting alpha parameter (influence of pheromone).
        beta_start (float): The starting beta parameter (influence of distance).
        alpha_end (float): The ending alpha parameter.
        beta_end (float): The ending beta parameter.

    Returns:
        Tuple[List[List[int]], float]: The best paths found and their length.
    """
    avg_weight = sum(graph[u][v]["weight"] for u, v in graph.edges) / len(graph.edges)
    k = avg_weight
    if not num_ants_start:
        num_ants_start = max(10, int(len(graph.nodes) / len(graph.edges) * 100))
    if not num_ants_end:
        num_ants_end = max(10, num_ants_start // 2)
    if not num_threads:
        num_threads = min(os.cpu_count() // 2, len(graph.nodes))
    if not dynamic_changes:
        dynamic_changes = {}

    pheromones = initialize_pheromones(graph, k)
    best_paths_set = set()
    best_path_length = float("inf")

    for iteration in range(num_iterations):
        if iteration in dynamic_changes:
            update_graph(graph, pheromones, dynamic_changes[iteration])

            if best_paths_set:
                valid_paths_set = set()
                for path in best_paths_set:
                    if is_path_valid(path, graph):
                        valid_paths_set.add(path)

                if valid_paths_set:
                    path_lengths = {
                        path: sum(
                            graph[path[i]][path[i + 1]]["weight"]
                            for i in range(len(path) - 1)
                        )
                        for path in valid_paths_set
                    }
                    min_length = min(path_lengths.values())
                    best_paths_set = {
                        path
                        for path, length in path_lengths.items()
                        if length == min_length
                    }
                    best_path_length = min_length
                else:
                    best_paths_set = set()
                    best_path_length = float("inf")

        num_ants = (
            num_ants_end
            + (num_ants_start - num_ants_end)
            * (num_iterations - iteration)
            / num_iterations
        )
        num_ants = int(num_ants)

        alpha, beta = adjust_parameters(
            iteration, num_iterations, alpha_start, beta_start, alpha_end, beta_end
        )

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(
                    simulate_ant, start_node, end_node, graph, pheromones, alpha, beta
                )
                for _ in range(num_ants)
            ]
            all_paths = [future.result() for future in futures if future.result()]

        valid_paths = [path for path in all_paths if is_path_valid(path, graph)]
        if valid_paths:
            pheromones = update_pheromones(
                pheromones,
                valid_paths,
                graph,
                iteration,
                num_iterations,
                min_rate,
                max_rate,
            )

        for path in valid_paths:
            length = sum(
                graph[path[i]][path[i + 1]]["weight"] for i in range(len(path) - 1)
            )
            path_tuple = tuple(path)
            if length < best_path_length:
                best_paths_set = {path_tuple}
                best_path_length = length
            elif length == best_path_length:
                best_paths_set.add(path_tuple)

        for path in list(best_paths_set):
            if not is_path_valid(path, graph):
                best_paths_set.remove(path)

    return list(best_paths_set), best_path_length


if __name__ == "__main__":

    def create_test_graph():
        """
        Creates a predefined graph for testing the ant colony optimization algorithm.

        Returns:
            nx.Graph: The graph with nodes and weighted edges.
        """
        graph = nx.Graph()
        edges = [
            (1, 2, 2),
            (1, 3, 5),
            (2, 3, 3),
            (2, 4, 4),
            (3, 4, 1),
            (4, 5, 2),
            (3, 5, 6),
        ]
        graph.add_weighted_edges_from(edges)
        return graph

    test_graph = create_test_graph()
    dynamic_changes = {10: [(1, 4, 7)], 20: [(2, 3, None)]}

    best_paths, best_path_length = run_ant_colony_dynamic(
        test_graph,
        num_iterations=30,
        start_node=1,
        end_node=5,
        dynamic_changes=dynamic_changes,
    )
    print("The best paths of graph 1:", best_paths)
    print("Length of the shortest path of graph 1:", best_path_length)

    def create_test_graph():
        """
        Creates a predefined graph for testing the ant colony optimization algorithm.

        Returns:
            nx.Graph: The graph with nodes and weighted edges.
        """
        graph = nx.Graph()
        edges = [
            (1, 2, 1),
            (1, 3, 2),
            (2, 4, 2),
            (3, 4, 1),
            (4, 5, 5),
            (3, 6, 8),
            (6, 5, 1),
            (5, 7, 2),
        ]
        for u, v, weight in edges:
            graph.add_edge(u, v, weight=weight)
        return graph

    graph = create_test_graph()

    dynamic_changes = {5: [(4, 5, 3), (3, 6, 10)], 10: [(2, 4, 1), (5, 7, 4)]}
    best_paths, best_path_length = run_ant_colony_dynamic(
        graph,
        num_iterations=50,
        start_node=1,
        end_node=5,
        dynamic_changes=dynamic_changes,
    )
    print("\nThe best paths of graph 2:", best_paths)
    print("Length of the shortest path of graph 2:", best_path_length)

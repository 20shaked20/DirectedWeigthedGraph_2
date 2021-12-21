import json
import math
import random
import sys
import time
from collections import deque
from typing import List

from Ex3.src.GraphInterface import GraphInterface
from Ex3.src.data.Project.DiGraph import DiGraph

INF = float("inf")


# START of TSP helper functions
def cost(x, y, m):
    """
    Calculates the cost between x and y using one of three cost methods
    :param x: a number to evaluate cost
    :param y: a number to evaluate cost
    :param m: dictates which cost algorithm to use ( 1 or 2 or 3 )
    :return: the cost between x and y
    """
    if x == y:
        return 0
    if m == 1:
        if x < 3 and y < 3:
            return 1
        if x < 3 or y < 3:
            return 200
        if x % 7 == y % 7:
            return 2
        return abs(x - y) + 3  # iff no other value was returned
    if m == 2:
        if x + y < 10:
            return abs(x - y) + 4
        if (x + y) % 11 == 0:
            return 3
        return abs(x - y) ** 2 + 10  # iff no other value was returned
    if m == 3:
        return (x + y) ** 2


def random_path(number_of_cities, seed):
    rnd_path = list(range(number_of_cities))
    random.seed(seed)
    random.shuffle(rnd_path)
    return rnd_path


def path_cost(cities, cost_function: int):
    """
    Given a list of cities calculates the cost of the path
    :param cities: a list of cities, their order in the list is a path
    :param cost_function: which cost function to use ( 1 or 2 or 3 )
    :return: the cost (or weight) of the path
    """
    total_cost = 0
    # cost_temp = 0
    n = len(cities)
    for i, city in enumerate(cities):
        if i == n - 1:
            cost_temp = cost(cities[i], cities[0], cost_function)
            total_cost += cost_temp
        else:
            cost_temp = cost(cities[i], cities[i + 1], cost_function)
            total_cost += cost_temp
    return total_cost


def mutation_operator(cities):
    """
    This mutation operator swaps two cities randomly to create a new path
    :param cities: a list of cities, their order in the list is a path
    :return: a list of cities, with two of them swapped randomly
    """
    # https://stackoverflow.com/questions/10623302/how-does-assignment-work-with-list-slices
    # https://www.geeksforgeeks.org/python-yield-keyword/
    r1 = list(range(len(cities)))
    r2 = list(range(len(cities)))
    random.shuffle(r1)
    random.shuffle(r2)
    for i in r1:
        for j in r2:
            if i < j:
                next_state = cities[:]  # copying the slice of cities into a variable called next
                next_state[i], next_state[j] = cities[j], cities[i]  # performs a swap into next and NOT in cities
                yield next_state


def probability_acceptance(prev_score, next_score, temperature):
    """
    This method calculates a probability using prev_score and next_score and the temperature of the SA algorithm
    :param prev_score: previous cost ( in this implementation )
    :param next_score: next cost ( in this implementation )
    :param temperature: the current temperature of the SA algorithm
    :return: a double representing a probability
    """
    if next_score < prev_score:
        return 1.0
    if temperature == 0:
        return 0.0
    return math.exp(-abs(next_score - prev_score) / temperature)


def cooling_schedule(start_temp, cooling_const):
    """
    This function continuously cools down the temperature of the SA algorithm using a constant value
    :param start_temp: the starting temperature of the SA algorithm
    :param cooling_const: a constant that satisfies 0 < cooling_const < 1
    :return: the value of the new temperature
    """
    t = start_temp
    while True:
        yield t
        t = cooling_const * t


def SA(number_of_cities, cost_function, MEB, seed):
    """
    This function implements a Simulated Annealing algorithm for TSP
    :param number_of_cities: self explanatory
    :param cost_function: which cost function to use ( 1 or 2 or 3 )
    :param MEB: Total number of evaluations allowed ( essentially, a limit on how much computational strength to use )
    :param seed: a seed for the random_path() generator
    :return: a tuple of (best path, best cost, number of evaluations)
    """
    # Note: these values are important for the offset of the SA algorithm and can be played around with
    start_temp = 70
    cooling_const = 0.9995
    # best_path = None
    # best_cost = None
    curr_path = random_path(number_of_cities, seed)
    curr_cost = path_cost(curr_path, cost_function)

    # if best_path is None or curr_cost < best_cost:
    best_cost = curr_cost
    best_path = curr_path

    evaluations = 1
    temp_schedule = cooling_schedule(start_temp, cooling_const)
    for temperature in temp_schedule:
        flag = False
        for next_path in mutation_operator(curr_path):
            if evaluations == MEB:
                flag = True
                break

            next_cost = path_cost(next_path, cost_function)

            if best_path is None or next_cost < best_cost:
                best_cost = next_cost
                best_path = next_path

            evaluations += 1
            p = probability_acceptance(curr_cost, next_cost, temperature)
            if random.random() < p:
                curr_path = next_path
                curr_cost = next_cost
                break
        if flag:
            break

    return best_path, best_cost, evaluations


# END of TSP helper functions


class GraphAlgo:

    def __init__(self, graph: GraphInterface):  # actually its DiGraph.
        """
        This is the constructor of the graph
        :param graph: represents a DiGraph.
        """
        self.graph = graph

    """This abstract class represents an interface of a graph."""

    def get_graph(self) -> GraphInterface:
        """
        :return: the directed graph on which the algorithm works on.
        """
        return self.graph

    def load_from_json(self, file_name: str) -> bool:
        """
        Loads a graph from a json file.
        @param file_name: The path to the json file
        @returns True if the loading was successful, False o.w.
        """
        try:
            with open(file_name, "r") as f:
                _dict = json.load(f)
                edges = _dict["Edges"]
                nodes = _dict["Nodes"]

                # NODE Creation:
                for k in nodes:
                    curr_pos = k["pos"]
                    xyz_split = curr_pos.split(",")
                    xyz_tuple = tuple(xyz_split)
                    curr_id = k["id"]
                    self.graph.add_node(curr_id, xyz_tuple)

                # EDGES Creation:
                for k in edges:
                    curr_src = k["src"]
                    curr_weight = k["w"]
                    curr_dest = k["dest"]
                    self.graph.add_edge(curr_src, curr_dest, curr_weight)
                return True
        except FileNotFoundError:
            return False

    def save_to_json(self, file_name: str) -> bool:
        """
        Saves the graph in JSON format to a file
        @param file_name: The path to the out file
        @return: True if the save was successful, False o.w.
        """
        raise NotImplementedError

    def shortest_path(self, id1: int, id2: int) -> (float, list):
        """
        Returns the shortest path from node id1 to node id2 using Dijkstra's Algorithm
        @param id1: The start node id
        @param id2: The end node id
        @return: The distance of the path, a list of the nodes ids that the path goes through
        Example:
        # >>> from GraphAlgo import GraphAlgo
        # >>> g_algo = GraphAlgo()
        # >>> g_algo.addNode(0)
        # >>> g_algo.addNode(1)
        # >>> g_algo.addNode(2)
        # >>> g_algo.addEdge(0,1,1)
        # >>> g_algo.addEdge(1,2,4)
        # >>> g_algo.shortestPath(0,1)
        # (1, [0, 1])
        # >>> g_algo.shortestPath(0,2)
        # (5, [0, 1, 2])
        Notes:
        If there is no path between id1 and id2, or one of them do not exist the function returns (float('inf'),[])
        More info:
        https://en.wikipedia.org/wiki/Dijkstra's_algorithm
        """
        if self.get_graph().get_all_v().get(id1) is None or self.get_graph().get_all_v().get(id2) is None:
            return INF, []
        unvisited_nodes = self.get_graph().get_all_v()

        # Creating a dictionary of each node's distance from start_node(id1).
        # it will be updated using relaxation on each traverse
        dist_from_id1 = {}
        for node in self.get_graph().get_all_v():
            if node == id1:
                dist_from_id1[node] = 0
            else:
                dist_from_id1[node] = INF

        # Initialize previous_node, the dictionary that maps each node to the
        # node it was visited from when the shortest path to it was found.
        previous_node = {node: None for node in self.get_graph().get_all_v()}

        while unvisited_nodes:
            # Set current_node to the unvisited node with the shortest distance
            # calculated so far.
            current_node = min(
                unvisited_nodes, key=lambda node: dist_from_id1[node]
            )
            unvisited_nodes.pop(current_node)

            # if there's a node that is not connected to our node, its value is INF.
            # no reason to keep checking because we can't traverse any further.
            if dist_from_id1[current_node] == INF:
                break

            # this is the relaxation process, checks if there's a shorter way to reach a neighbour,
            for k in self.get_graph().all_out_edges_of_node(current_node):
                distance = self.get_graph().all_out_edges_of_node(current_node)[k]
                neighbor = k
                new_path = dist_from_id1[current_node] + distance
                if new_path < dist_from_id1[neighbor]:
                    dist_from_id1[neighbor] = new_path
                    previous_node[neighbor] = current_node

            if current_node == id2:  # src = dest, we reached the end, finished with loop.
                break

        # To build the path to be returned,
        # we iterate back from the nodes, to get the path. using likewise "parent array"
        path = deque()
        current_node = id2
        while previous_node[current_node] is not None:
            path.appendleft(current_node)
            current_node = previous_node[current_node]
        path.appendleft(id1)

        if dist_from_id1[id2] is INF or dist_from_id1[id2] <= 0:
            return INF, []

        return path, dist_from_id1[id2]

    def TSP(self, node_lst: List[int]) -> (List[int], float):
        """
        Finds the shortest path that visits all the nodes in the list
        :param node_lst: A list of nodes id's
        :return: A list of the node id's in the path, and the overall distance (or cost/weight, etc...)
        """
        # all helper functions are static
        number_of_cities = len(node_lst)
        MEB = 200000  # this is the maximum number of evaluations allowed in our function - IMPORTANT!
        cost_function = 1  # TODO: test all cost function to see which one works best with our graphs ( 1 or 2 or 3 )
        seed = 26  # this is a seed for a random path generator ( see SA implementation )
        # start_time = time.time()
        best_path, best_cost, evaluations = SA(number_of_cities, cost_function, MEB, seed)
        # end_time = time.time()
        return best_path, best_cost

    def centerPoint(self) -> (int, float):
        """
        Finds the node that has the shortest distance to it's farthest node.
        :return: The node id, min-maximum distance
        """
        min1 = sys.float_info.max
        node_id = -1
        for k in self.get_graph().get_all_v():
            curr_node = k
            max1 = sys.float_info.min
            for v in self.get_graph().get_all_v():
                if v == k:  # same node, no need to check again.
                    continue
                next_node = v
                tmp_dijk = self.shortest_path(curr_node, next_node)
                tmp = tmp_dijk[1]
                if tmp_dijk[0] is not INF:
                    if tmp > max1:
                        max1 = tmp
                    if tmp > min1:
                        # we want Minimum of all maximums
                        break

            if min1 > max1:
                min1 = max1
                node_id = k

        return node_id, min1

    def plot_graph(self) -> None:
        """
        Plots the graph.
        If the nodes have a position, the nodes will be placed there.
        Otherwise, they will be placed in a random but elegant manner.
        @return: None
        """
        raise NotImplementedError


if __name__ == '__main__':
    nodes = {}
    tmp = DiGraph(nodes)
    g = GraphAlgo(tmp)
    g.load_from_json("/Users/Shaked/PycharmProjects/DirectedWeigthedGraph_2/Ex3/data/1000Nodes.json")
    # print(g.get_graph().all_out_edges_of_node(0))
    # print(g.get_graph().all_out_edges_of_node(0)[5])
    print(g.shortest_path(0, 6))
    print(g.centerPoint())

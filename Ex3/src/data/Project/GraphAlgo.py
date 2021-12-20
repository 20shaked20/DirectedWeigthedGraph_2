import json
import sys
from collections import deque
from Ex3.src.GraphInterface import GraphInterface
from Ex3.src.data.Project.DiGraph import DiGraph
from typing import List

INF = float("inf")


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
        If there is no path between id1 and id2, or one of them dose not exist the function returns (float('inf'),[])
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
        # node it was visited from when the the shortest path to it was found.
        previous_node = {node: None for node in self.get_graph().get_all_v()}

        while unvisited_nodes:
            # Set current_node to the unvisited node with shortest distance
            # calculated so far.
            current_node = min(
                unvisited_nodes, key=lambda node: dist_from_id1[node]
            )
            unvisited_nodes.pop(current_node)

            # if there's a node that is not connected to our node, its value is INF.
            # no reason to keep checking because we cant traverse any further.
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
        :return: A list of the nodes id's in the path, and the overall distance
        """

    def centerPoint(self) -> (int, float):
        """
        Finds the node that has the shortest distance to it's farthest node.
        :return: The nodes id, min-maximum distance
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
                    if tmp > min1:  # in case there's a bigger dist break, because we don't want that -> Minimum of all maximum
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

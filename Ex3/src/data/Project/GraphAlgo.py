import json
import time
import sys
from collections import deque
from typing import List

# Drawing imports
import tkinter
import matplotlib.pyplot as plt
from matplotlib.widgets import *
from numpy.random import uniform
from tkinter import *
from tkinter import filedialog as fd
from tkinter.filedialog import asksaveasfile
#

from Ex3.src.GraphInterface import GraphInterface
from Ex3.src.data.Project.DiGraph import DiGraph

INF = float("inf")

class GraphAlgo:

    def __init__(self, graph: GraphInterface = None):  # actually its DiGraph.
        """
        This is the constructor of the graph
        :param graph: represents a DiGraph.
        """
        if graph is None:
            self.graph = DiGraph()
        else:
            self.graph = graph
        self.nodesX = {}
        self.nodesY = {}

        self.ax = None
        self.root = Tk()

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
                xyz_tuple = None  # standard position
                key = "pos"
                for k in nodes:
                    if key in k:
                        curr_pos = k["pos"]
                        xyz_split = curr_pos.split(",")
                        xyz_tuple = tuple(xyz_split)
                        curr_id = k["id"]
                        self.graph.add_node(curr_id, xyz_tuple)
                    else:
                        xyz_tuple = (uniform(0.0, 100.0), uniform(0.0, 100.0), 0.0)  # in case it has
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
        data = {"Edges": [], "Nodes": []}
        all_nodes = self.get_graph().get_all_v()
        for node in all_nodes:
            # key_list = list(all_nodes.keys())
            val_list = list(all_nodes.values())
            # print(val_list[node][0], ',' + val_list[node][1], ',', val_list[node][2])
            pos = str(val_list[node][0]) + ',' + str(val_list[node][1]) + ',' + str(val_list[node][2])
            data["Nodes"].append({
                'pos': pos,
                'id': node
            })
            # Iterate once over OUT-going edges
            edges = self.get_graph().all_out_edges_of_node(node)
            key_list = list(edges.keys())
            val_list = list(edges.values())
            for edge in edges:
                if not (edge in data["Edges"]):
                    data["Edges"].append({
                        'src': node,
                        'w': val_list[key_list.index(edge)],
                        'dest': edge
                    })

            # Then iterate once over IN-going edges
            edges = self.get_graph().all_in_edges_of_node(node)
            key_list = list(edges.keys())
            val_list = list(edges.values())
            for edge in edges:
                if not (edge in data["Edges"]):
                    data["Edges"].append({
                        'src': edge,
                        'w': val_list[key_list.index(edge)],
                        'dest': node
                    })
        # open(file_name, "x")  # cant "wx" - meaning create (x) + write (w), but just "w" will create if needed!
        json_file = open(str(file_name), "w")
        json_file.write(json.dumps(data, indent=4, ensure_ascii=False))
        return True
        """
        Explanation on values passed to json.dumps():
        indent = 4  -  activates pretty print if an integer is passed
        sort_keys = True  -  sorts the json 
        ensure_ascii = False  -  does not escape unicode characters to match ASCII 
        ( for example 'cost':'£4.00' becomes --> "cost": "\u00a34.00" if ensure_ascii=True )
        """

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

        return list(path), dist_from_id1[id2]

    def TSP(self, node_lst: List[int]) -> (List[int], float):
        """
        Finds the shortest path that visits all the nodes in the list
        :param node_lst: A list of nodes id's
        :return: A list of the node id's in the path, and the overall distance (or cost/weight, etc...)
        """
        start_time = time.time()
        best_path = self.greedy_tsp(node_lst)
        if best_path is None:
            return None, -1
        temp_path = copy.deepcopy(best_path)
        best_cost = self.path_weight(temp_path)
        end_time = time.time()
        # print("Approximated a solution for TSP in:", end_time - start_time, "seconds with weight:", best_cost)
        return best_path, best_cost
        # return self.fixPath(node_lst, best_path), best_cost

    """
    START of TSP helper functions
    """

    def cost(self, node, neighbour):
        if node == neighbour:
            return 0.0
        neighbours = self.get_graph().all_out_edges_of_node(node)
        if len(neighbours) > 0:
            return neighbours.get(neighbour)
        return sys.float_info.max

    def path_weight(self, path: list[int]):
        path.reverse()
        ans = 0
        prev = path.pop()
        while path:
            next_n = path.pop()
            temp = self.cost(prev, next_n)
            ans += temp
            prev = next_n
        return ans

    def get_cheapest_neighbour(self, node):
        """
        This method find the cheapest neighbour of a node, and returns it.
        :param node: An integer for a node for which we want the cheapest neighbour
        :return: An int representing the cheapest neighbour node id, or -1 if no neighbours
        """
        neighbours = self.get_graph().all_out_edges_of_node(node)
        if len(neighbours) == 0:
            return -1
        best_cost = sys.float_info.max
        for neighbour in neighbours:
            cost = self.cost(node, neighbour)
            if cost < best_cost:
                best_cost = cost
                best_neighbour = neighbour
        return best_neighbour

    def greedy_tsp(self, cities: list[int]) -> List[int]:
        """
        This method heuristically computes a path for a variation on the TSP problem ( can repeat visits)
        :param cities: a list of nodes to visit
        :return: an ordered path of nodes representing a heuristically best route
        """
        curr_node = cities.pop()
        path = [curr_node]

        while len(cities) > 0:

            if len(cities) == 1:
                last_city = cities.pop()
                temp = self.shortest_path(path[-1], last_city)
                if temp[0] == INF:
                    return None
                path_to_rel = tuple(temp)
                path_to_rel = path_to_rel[0]
                path_to_rel.reverse()
                path_to_rel.pop()
                while path_to_rel:
                    t = path_to_rel.pop()
                    path.append(t)
                    if t in cities:
                        cities.remove(t)
                    if len(cities) == 0:
                        break
                # to_visit.remove(min(to_visit))
                continue  # fixed being stuck by plotting a path to a point in the list

            try:
                curr_node = self.get_cheapest_neighbour(path[-1])  # arr[-1] accesses last member of array
            except:
                curr_node = -1

            if curr_node in cities:
                cities.remove(curr_node)

            # exceptions : same solution, seperated for semantic reasons ( i.e. accessing path[-2] )

            # 1. no neighbours ( stuck )

            if curr_node == -1:
                temp = self.shortest_path(path[-1], cities.pop())
                if temp[0] == INF:
                    return None
                path_to_rel = tuple(temp)
                path_to_rel = path_to_rel[0]
                path_to_rel.reverse()
                path_to_rel.pop()
                while path_to_rel:
                    t = path_to_rel.pop()
                    path.append(t)
                    if t in cities:
                        cities.remove(t)
                    if len(cities) == 0:
                        break
                # to_visit.remove(min(to_visit))
                continue  # fixed being stuck by plotting a path to a point in the list

            # 2. looping ( repeating the same path )

            if len(path) >= 2:
                if path[-2] == curr_node:  # if we detect a loop, or we are stuck
                    # path_to_rel = self.shortest_path(path[-1], cities.pop())
                    temp = self.shortest_path(path[-1], cities.pop())
                    if temp[0] == INF:
                        return None
                    path_to_rel = tuple(temp)
                    path_to_rel: list = path_to_rel[0]  # this is a deque object
                    path_to_rel.reverse()
                    path_to_rel.pop()
                    while path_to_rel:
                        t = path_to_rel.pop()
                        path.append(t)
                        if t in cities:
                            cities.remove(t)
                        if len(cities) == 0:
                            break
                    # to_visit.remove(min(to_visit))
                    continue  # fixed being stuck by plotting a path to a point in the list

            # if all is 'right'

            path.append(curr_node)
        return path

    """
    END of TSP helper functions
    """

    def is_connected(self) -> bool:
        """
        private method that checks if the graph is strongly connected or not.
        :return: True if yes, false if not.
        """
        all_nodes = self.get_graph().get_all_v()
        for node in all_nodes:
            if not self.BFS_Helper(node):
                return False
        return True

    def BFS_Helper(self, node: int) -> bool:
        visited = {node}
        q = [node]
        while q:
            temp = q.pop()
            edges = self.get_graph().all_out_edges_of_node(temp)
            for edge in edges:
                if edge not in visited:
                    visited.add(edge)
                    q.append(edge)
        return len(visited) == len(self.get_graph().get_all_v())

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

    """
    From here an on all the methods relate to plot graph.
    """

    def plot_graph(self) -> None:
        """
        Plots the graph.
        If the nodes have a position, the nodes will be placed there.
        Otherwise, they will be placed in a random but elegant manner.
        @return: None
        """
        self.root.title("Graph Menu")
        self.root.geometry("250x150")
        self.root.eval('tk::PlaceWindow . center')
        self.root.resizable(False, False)
        show_graph = tkinter.Button(self.root, text="show graph", command=self.root_init, fg='black')
        load_graph = tkinter.Button(self.root, text="load graph", command=self.load_graph, fg='blue')
        save_graph = tkinter.Button(self.root, text="save graph", command=self.save_graph, fg='blue')
        show_graph.pack()
        load_graph.pack()
        save_graph.pack()
        self.root.mainloop()

    def root_init(self) -> None:
        fig = plt.figure(figsize=(20, 8), facecolor='gray', edgecolor='blue')
        self.ax = fig.subplots()  # for graph
        plt.subplots_adjust(left=0.2, bottom=0.1)
        self.update_x_y_pos()
        for k in self.nodesX:
            plt.plot(self.nodesX.get(k), self.nodesY.get(k), markersize=6, marker='o', color='blue')  # NODE
            plt.text(self.nodesX.get(k) + 10, self.nodesY.get(k) + 5, str(k), color='black', fontsize=10)  # ID
        self.draw_line_arrows()
        plt.title('Directed Weighted Graph')
        plt.ylabel('y-axis')
        plt.xlabel('x-axis')
        # button functionality

        # CENTER:
        c_loc = plt.axes([0.04, 0.60, 0.1, 0.075])
        c_b = plt.Button(c_loc, label='Center', hovercolor='blue')
        c_b.on_clicked(self.draw_center)

        # DIJKSTRA
        box_loc = plt.axes([0.04, 0.80, 0.1, 0.075])
        text_box = TextBox(box_loc, 'Dijkstra', initial="Insert id1,id2")
        text_box.on_submit(self.draw_sp)

        # TSP
        tsp_loc = plt.axes([0.04, 0.70, 0.1, 0.075])
        tsp_box = TextBox(tsp_loc, 'TSP', initial="Insert cities")
        tsp_box.on_submit(self.draw_cities)

        # Clear
        clr_loc = plt.axes([0.04, 0.50, 0.1, 0.075])
        clr_but = plt.Button(clr_loc, label='Clear', hovercolor='blue')
        clr_but.on_clicked(self.clear)

        # Remove Node
        rm_node = plt.axes([0.04, 0.40, 0.1, 0.075])
        rm_node_box = TextBox(rm_node, ' Remove\nNode', initial="Insert id")
        rm_node_box.on_submit(self.remove_node)

        # Remove Edge
        rm_edge = plt.axes([0.04, 0.30, 0.1, 0.075])
        rm_edge_box = TextBox(rm_edge, ' Remove\nEdge', initial="Insert id1,id2")
        rm_edge_box.on_submit(self.remove_edge)

        # Add Node
        add_node = plt.axes([0.04, 0.20, 0.1, 0.075])
        add_node_box = TextBox(add_node, ' Add\nNode', initial="Insert id,x,y")
        add_node_box.on_submit(self.add_node)

        # Add Edge
        add_edge = plt.axes([0.04, 0.10, 0.1, 0.075])
        add_edge_box = TextBox(add_edge, ' Add\nEdge', initial="Insert id1,id2,w")
        add_edge_box.on_submit(self.add_edge)

        plt.show()

        while plt.fignum_exists(1):
            self.root.withdraw()

        self.root.deiconify()

    def load_graph(self) -> None:
        filetypes = (
            ('json files', '*.json'),
            ('All files', '*.*')
        )
        filename = fd.askopenfilename(title='Open a file',
                                      initialdir='/../../PycharmProjects/DirectedWeigthedGraph_2/Ex3/data',
                                      filetypes=filetypes)
        self.load_from_json(filename)

    def save_graph(self) -> None:
        filetypes = (
            ('json files', '*.json'),
            ('All files', '*.*')
        )
        filename = fd.asksaveasfile(title="Save a file",
                                    initialdir='/../../PycharmProjects/DirectedWeigthedGraph_2/Ex3/data',
                                    filetypes=filetypes)
        self.save_to_json(filename.name)

    def draw_line_arrows(self) -> None:
        """
        private method to draw arrow lines for the graph.
        """
        nodes = self.get_graph().get_all_v()
        for k in nodes:
            out_edge = self.get_graph().all_out_edges_of_node(k)
            start = [self.nodesX[k], self.nodesY[k]]
            for v in out_edge:
                end = [self.nodesX[v], self.nodesY[v]]
                plt.annotate("", xy=(end[0], end[1]), xytext=(start[0], start[1]), arrowprops=dict(arrowstyle="->")) # reversed position

    def draw_center(self, event) -> None:
        center = self.centerPoint()[0]
        self.ax.plot(self.nodesX[center], self.nodesY[center], markersize=6, marker='o', color='magenta')
        plt.show()

    def draw_sp(self, event) -> None:
        data = eval(event)
        id1 = data[0]
        id2 = data[1]
        path = list((self.shortest_path(id1, id2)[0]))
        # print(path)
        i = 0
        for k in range(0, len(path)):
            start = (self.nodesX[path[i]], self.nodesY[path.pop(i)])
            end = (self.nodesX[path[i]], self.nodesY[path[i]])
            self.ax.annotate("", xy=(end[0], end[1]), xytext=(start[0], start[1]),
                             arrowprops=dict(arrowstyle="fancy"))  # reversed position here because of arrow location
        plt.show()

    def draw_cities(self, event) -> None:
        data = list(eval(event))
        path = list(self.TSP(data)[0])
        path.reverse()
        i = 0
        print(path)
        for k in range(0, len(path)):
            start = (self.nodesX[path[i]], self.nodesY[path.pop(i)])
            end = (self.nodesX[path[i]], self.nodesY[path[i]])
            self.ax.annotate("", xy=(end[0], end[1]), xytext=(start[0], start[1]),
                             arrowprops=dict(arrowstyle="fancy"))  # reversed position here because of arrow location
        plt.show()

    def remove_node(self, event) -> None:
        data = eval(event)
        self.graph.remove_node(data)
        plt.close()
        self.root_init()

    def remove_edge(self, event) -> None:
        data = eval(event)
        self.graph.remove_edge(data[0], data[1])
        plt.close()
        self.root_init()

    def add_node(self, event) -> None:
        data = eval(event)
        pos = (data[1], data[2], 0)
        self.graph.add_node(data, pos)
        plt.close()
        self.root_init()

    def add_edge(self, event) -> None:
        data = eval(event)
        self.graph.add_edge(data[0], data[1], data[2])
        plt.close()
        self.root_init()

    def update_x_y_pos(self) -> None:
        """
        private method that updates the positions of the nodes to fit correctly in the plot
        """
        self.nodesY = {}
        self.nodesX = {}

        minX = sys.float_info.max
        minY = sys.float_info.max
        maxX = sys.float_info.min
        maxY = sys.float_info.min

        nodes = self.get_graph().get_all_v()

        for k in nodes:
            curr_pos = nodes[k]
            x = float(curr_pos[0])
            y = float(curr_pos[1])
            minX = min(minX, x)
            minY = min(minY, y)
            maxX = max(maxX, x)
            maxY = max(maxY, y)

        X_Scaling = 1000 / (maxX - minX) * 0.9
        Y_Scaling = 1000 / (maxY - minY) * 0.9

        for k in nodes:
            curr_id = k
            curr_pos = nodes[k]
            x = (float(curr_pos[0]) - minX) * X_Scaling + 10
            y = (float(curr_pos[1]) - minY) * Y_Scaling + 30

            self.nodesX[curr_id] = int(x)
            self.nodesY[curr_id] = int(y)

    def clear(self, event) -> None:
        plt.close()
        self.root_init()


if __name__ == '__main__':
    nodes = {}
    nodes = {}
    tmp = DiGraph(nodes)
    g = GraphAlgo(tmp)
    # g.load_from_json("/Users/Shaked/PycharmProjects/DirectedWeigthedGraph_2/Ex3/data/A5.json")
    g.load_from_json("C:/Users/yonar/PycharmProjects/DirectedWeigthedGraph_2/Ex3/data/A5.json")
    # print(g.get_graph().all_out_edges_of_node(0))
    # print(g.get_graph().all_out_edges_of_node(0)[5])
    # print(g.shortest_path(5, 10))
    # print(g.centerPoint())
    g.plot_graph()

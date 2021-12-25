import copy
from unittest import TestCase
from Ex3.src.GraphInterface import GraphInterface
from Ex3.src.data.Project.DiGraph import DiGraph
from Ex3.src.data.Project.GraphAlgo import GraphAlgo
from Ex3.src.data.Project.Node import Node
from typing import List
import os


class TestGraphAlgo(TestCase):

    def setUp(self) -> None:
        nodes = {}
        g = DiGraph(nodes)
        path = "/Users/Shaked/PycharmProjects/DirectedWeigthedGraph_2/Ex3/data/100Nodes.json"
        # path = "C:/Users/yonar/PycharmProjects/DirectedWeigthedGraph_2/Ex3/data/A5.json"
        self.graph_algo = GraphAlgo(g)
        self.graph_algo.load_from_json(path)

    def test_get_graph(self):
        tmp_DiGraph = self.graph_algo.get_graph()
        self.assertEqual(self.graph_algo.get_graph(), tmp_DiGraph)

    def test_load_from_json(self):
        file_loc = "/Users/Shaked/PycharmProjects/DirectedWeigthedGraph_2/Ex3/data/100Nodes.json"
        # file_loc = "C:/Users/yonar/PycharmProjects/DirectedWeigthedGraph_2/Ex3/data/A5.json"
        self.assertEqual(self.graph_algo.load_from_json(file_loc), True)
        self.assertEqual(self.graph_algo.load_from_json("bla"), False)

    def test_save_to_json(self):
        # file_loc = "/Users/Shaked/PycharmProjects/DirectedWeigthedGraph_2/Ex3/data/A0.json"
        file_loc = "C:/Users/yonar/PycharmProjects/DirectedWeigthedGraph_2/Ex3/data/A5.json"
        self.graph_algo.load_from_json(file_loc)
        self.graph_algo.save_to_json("test_save")

    def test_shortest_path(self):
        print(self.graph_algo.shortest_path(0, 6))

    def test_tsp(self):
        cities = {3, 2, 14, 5, 11, 10, 4}
        check = copy.deepcopy(cities)
        tsp = self.graph_algo.TSP(cities)[0]
        print(tsp)

    def test_is_connected(self):
        self.assertTrue(self.graph_algo.is_connected())

    def test_center_point(self):
        print(self.graph_algo.centerPoint())

    def test_plot_graph(self):
        self.graph_algo.plot_graph()

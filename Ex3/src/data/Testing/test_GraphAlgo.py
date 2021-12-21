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
        # path = "/Users/Shaked/PycharmProjects/DirectedWeigthedGraph_2/Ex3/data/A5.json"
        path = "C:/Users/yonar/PycharmProjects/DirectedWeigthedGraph_2/Ex3/data/A5.json"
        self.graph_algo = GraphAlgo(g)
        self.graph_algo.load_from_json(path)

    def test_get_graph(self):
        tmp_DiGraph = self.graph_algo.get_graph()
        self.assertEqual(self.graph_algo.get_graph(), tmp_DiGraph)

    def test_load_from_json(self):
        # file_loc = "/Users/Shaked/PycharmProjects/DirectedWeigthedGraph_2/Ex3/data/A0.json"
        file_loc = "C:/Users/yonar/PycharmProjects/DirectedWeigthedGraph_2/Ex3/data/A5.json"
        self.assertEqual(self.graph_algo.load_from_json(file_loc), True)
        self.assertEqual(self.graph_algo.load_from_json("bla"), False)

    def test_save_to_json(self):
        self.fail()

    def test_shortest_path(self):
        self.fail()

    def test_tsp(self):
        self.fail()

    def test_center_point(self):
        print(self.graph_algo.centerPoint())

    def test_plot_graph(self):
        self.fail()

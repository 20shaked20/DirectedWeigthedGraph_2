from unittest import TestCase
from Ex3.src.data.Project.Node import Node
from Ex3.src.data.Project.DiGraph import DiGraph


class TestDiGraph(TestCase):
    def setUp(self) -> None:
        pos0 = (35.18753053591606, 32.10378225882353, 0.0)
        self.node0 = Node(0, pos0)
        pos1 = (35.18958953510896, 32.10785303529412, 0.0)
        self.node1 = Node(1, pos1)
        pos2 = (35.19341035835351, 32.10610841680672, 0.0)
        self.node2 = Node(2, pos2)
        nodes = {0: self.node0, 1: self.node1, 2: self.node2}
        self.g = DiGraph(nodes)

    def test_v_size(self):
        self.assertEqual(self.g.v_size(), 3)

    def test_e_size(self):
        self.assertEqual(self.g.e_size(), 0)

    def test_get_all_v(self):
        tmp_dict = {0: (35.18753053591606, 32.10378225882353, 0.0),
                    1: (35.18958953510896, 32.10785303529412, 0.0),
                    2: (35.19341035835351, 32.10610841680672, 0.0)}
        print(self.g.get_all_v())
        self.assertEqual(self.g.get_all_v(), tmp_dict)

    def test_all_in_edges_of_node(self):
        tmp_all_in_e_n0 = {1: 5.5,
                           2: 5.5}
        weight = 5.5
        self.node0.add_edge_in(1, weight)  # adding edge : 0 <--- 1 with weight 5.5
        self.node0.add_edge_in(2, weight)  # adding edge : 0 <--- 2 with weight 5.5
        self.assertEqual(self.g.all_in_edges_of_node(0), tmp_all_in_e_n0)

    def test_all_out_edges_of_node(self):
        tmp_all_out_e_n0 = {1: 5.5,
                            2: 5.5}
        weight = 5.5
        self.node0.add_edge_out(1, weight)  # adding edge : 0 ---> 1 with weight 5.5
        self.node0.add_edge_out(2, weight)  # adding edge : 0 ---> 2 with weight 5.5
        self.assertEqual(self.g.all_out_edges_of_node(0), tmp_all_out_e_n0)

    def test_get_mc(self):
        self.assertEqual(self.g.get_mc(), 0)

    def test_add_edge(self):
        self.fail()

    def test_add_node(self):
        tmp_dict = {0: (35.18753053591606, 32.10378225882353, 0.0),
                    1: (35.18958953510896, 32.10785303529412, 0.0),
                    2: (35.19341035835351, 32.10610841680672, 0.0),
                    3: (5, 5, 5)}
        pos = (5, 5, 5)
        self.assertEqual(self.g.add_node(3, pos), True)
        self.assertEqual(self.g.get_all_v(), tmp_dict)  # checking it was added.
        self.g.add_node(4, pos)
        self.assertEqual(self.g.add_node(4, pos), False)

    def test_remove_node(self):
        self.fail()

    def test_remove_edge(self):
        self.fail()

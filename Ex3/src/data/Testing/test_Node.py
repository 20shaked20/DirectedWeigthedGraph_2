from unittest import TestCase
from Ex3.src.data.Project.Node import Node


class TestNode(TestCase):

    def setUp(self) -> None:
        pos0 = (35.18753053591606, 32.10378225882353, 0.0)
        self.node0 = Node(0, pos0)
        pos1 = (35.18958953510896, 32.10785303529412, 0.0)
        self.node1 = Node(1, pos1)
        pos2 = (35.19341035835351, 32.10610841680672, 0.0)
        self.node2 = Node(2, pos2)

    def test_get_id(self):
        self.assertEqual(self.node0.get_id(), 0)
        self.assertEqual(self.node1.get_id(), 1)

    def test_get_pos(self):
        self.assertEqual(self.node0.get_pos(), (35.18753053591606, 32.10378225882353, 0.0))
        self.assertEqual(self.node1.get_pos(), (35.18958953510896, 32.10785303529412, 0.0))

    def test_set_pos(self):
        new_pos = (5, 5, 0)
        self.node0.set_pos(new_pos)
        self.assertEqual(self.node0.get_pos(), (5, 5, 0))

    def test_add_edge_in(self):
        weight = 5.5
        self.node0.add_edge_in(1, weight)  # adding edge : 0 <--- 1 with weight 5.5
        self.assertEqual(self.node0.edges_in.get(1), weight)

    def test_add_edge_out(self):
        weight = 5.5
        self.node1.add_edge_out(0, weight)  # adding edge : 1 ----> 0 with weight 5.5
        self.assertEqual(self.node1.edges_out.get(0), weight)

    def test_node_dict(self):
        check_dict = {'pos': "(35.18753053591606, 32.10378225882353, 0.0)",
                      'id': 0}
        self.assertEqual(self.node0.node_dict(), check_dict)

    def test_all_edges_out_dict(self):
        # Node2 out edges dict : ( 2 ---> dest )
        weight = 5.5
        self.node2.add_edge_out(0, weight)
        self.node2.add_edge_out(1, weight)
        check_dict = {0: 5.5,
                      1: 5.5}
        self.assertEqual(self.node2.all_edges_out_dict(), check_dict)

    def test_all_edges_in_dict(self):
        # Node2 in edges dict : ( 2 <--- src )
        weight = 5.5
        self.node2.add_edge_in(0, weight)
        self.node2.add_edge_in(1, weight)
        check_dict = {0: 5.5,
                      1: 5.5}
        self.assertEqual(self.node2.all_edges_in_dict(), check_dict)

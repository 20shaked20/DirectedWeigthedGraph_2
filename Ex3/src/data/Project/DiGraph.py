from Ex3.src.data.Project.Node import Node


class DiGraph:

    def __init__(self, nodes: dict) -> None:
        """
        Constructor method, using a node dictionary to store all nodes & their edges.
        :param nodes: dictionary of nodes
        MC: mode counter for graph changes ( also known as graph version ) : integer
        node_size : number of nodes in the graph. : integer
        edges_size : number of edges in the graph ->
        -> is set to zero because at first there are only edges going out of nodes so we cant tell if they are connected
        """
        self.nodes = nodes
        self.MC = 0
        self.node_size = len(nodes)  # how many nodes are in the graph.
        self.edges_size = 0  # right now there are 0 edges, only nodes.

    def v_size(self) -> int:
        """
        Returns the number of vertices in this graph
        @return: The number of vertices in this graph
        """
        return len(self.nodes)

    def e_size(self) -> int:
        """
        Returns the number of edges in this graph
        @return: The number of edges in this graph
        """
        return self.edges_size

    def get_all_v(self) -> dict:
        """return a dictionary of all the nodes in the Graph, each node is represented using a pair
         (node_id, node_data)
        """
        all_v_dict = {}
        for k in self.nodes.keys():
            all_v_dict[k] = self.nodes.get(k).get_pos()
        return all_v_dict

    def all_in_edges_of_node(self, id1: int) -> dict:
        """return a dictionary of all the nodes connected to (into) node_id ,
        each node is represented using a pair (other_node_id, weight)
         """
        return self.nodes.get(id1).all_edges_in_dict()

    def all_out_edges_of_node(self, id1: int) -> dict:
        """return a dictionary of all the nodes connected from node_id , each node is represented using a pair
        (other_node_id, weight)
        """
        return self.nodes.get(id1).all_edges_out_dict()

    def get_mc(self) -> int:
        """
        Returns the current version of this graph,
        on every change in the graph state - the MC should be increased
        @return: The current version of this graph.
        """
        return self.MC

    def add_edge(self, id1: int, id2: int, weight: float) -> bool:
        """
        Adds an edge to the graph.
        @param id1: The start node of the edge
        @param id2: The end node of the edge
        @param weight: The weight of the edge
        @return: True if the edge was added successfully, False o.w.
        Note: If the edge already exists or one of the nodes dose not exists the functions will do nothing
        """
        if id1 in self.nodes and id2 in self.nodes:  # Both nodes exist
            self.nodes.get(id1).add_edge_out(weight)  # connects out id1 ----> id2
            self.nodes.get(id2).add_edge_in(weight)  # connects in id1 ----> id2
        return False

    def add_node(self, node_id: int, pos: tuple = None) -> bool:
        """
        Adds a node to the graph.
        @param node_id: The node ID
        @param pos: The position of the node
        @return: True if the node was added successfully, False o.w.
        Note: if the node id already exists the node will not be added
        """
        if node_id in self.nodes:
            return False
        tmp_node = Node(node_id, pos)
        self.nodes[node_id] = tmp_node
        self.MC += 1
        return True

    def remove_node(self, node_id: int) -> bool:
        """
        Removes a node from the graph.
        @param node_id: The node ID
        @return: True if the node was removed successfully, False o.w.
        Note: if the node id does not exists the function will do nothing
        """
        raise NotImplementedError

    def remove_edge(self, node_id1: int, node_id2: int) -> bool:
        """
        Removes an edge from the graph.
        @param node_id1: The start node of the edge
        @param node_id2: The end node of the edge
        @return: True if the edge was removed successfully, False o.w.
        Note: If such an edge does not exists the function will do nothing
        """
        raise NotImplementedError

from Ex3.src.data.Project.Node import Node


# TODO:
#  1. remove node
#  2. remove edge
#  3. finish tests

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
        self.nodes = {}
        self.MC = 0
        self.node_size = len(nodes)
        self.edges_size = 0  # adding edges later, so it's set to 0

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
        if self.nodes.get(id1) is None or self.nodes.get(id2) is None:  # one of nodes are not available at all
            return False
        if id2 in self.nodes.get(id1).all_edges_out_dict().keys():  # edge exist
            return False
        if id1 == id2:  # same id cannot have own edge
            return False
        self.nodes.get(id1).add_edge_out(id2, weight)
        self.nodes.get(id2).add_edge_in(id1, weight)
        self.edges_size += 1
        self.MC += 1
        return True

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
        TODO: make this work.
        Removes a node from the graph.
        @param node_id: The node ID
        @return: True if the node was removed successfully, False o.w.
        Note: if the node id does not exists the function will do nothing
        """
        if node_id not in self.nodes:  # node does not exist case.
            return False
        pass

    def remove_edge(self, node_id1: int, node_id2: int) -> bool:
        """
        TODO: make this work.
        Removes an edge from the graph.
        @param node_id1: The start node of the edge
        @param node_id2: The end node of the edge
        @return: True if the edge was removed successfully, False o.w.
        Note: If such an edge does not exists the function will do nothing
        """
        if node_id1 == node_id2:  # same id can't add that edge at all.
            return False
        if node_id1 in self.nodes.get(node_id1).all_edges_out_dict().keys():  # if edge exists
            pass

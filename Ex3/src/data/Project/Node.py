class Node:

    def __init__(self, _id: int, _pos: tuple) -> None:
        """
        :param _id: integer represents a node id.
        :param _pos: a tuple represent x,y,z of a node. ( z is always 0 )
        _edges_in: dictionary of edges going out of the node.
        _edges_out: dictionary of edges going into the node.
        """
        self._id = _id
        self._pos = _pos
        self.edges_in = {}
        self.edges_out = {}

    def get_id(self) -> int:
        """
        :return: id of the node.
        """
        return self._id

    def get_pos(self) -> tuple:
        """
        :return: position of the node (x,y,z)
        """
        return self._pos

    def set_pos(self, new_pos: tuple) -> None:
        """
        Updates a position of a node.
        :param new_pos: a tuple represent x,y,z of a node. ( z is always 0 )
        """
        self._pos = new_pos

    def add_edge_in(self, from_node: int, weight: float) -> None:
        """
        Adds an edge from another node to this node. ( like this :  13 <------ 4 )
        :param from_node: id represent the src node we getting the edge from.
        :param weight: weight of the added edge.
        """
        self.edges_in[from_node] = weight

    def add_edge_out(self, to_node: int, weight: float) -> None:
        """
        Adds an edge from this node to another node ( like this : 13 ------> 4 )
        :param to_node: id represent the src node we getting the edge from.
        :param weight: weight of the added edge.
        """
        self.edges_out[to_node] = weight

    def node_dict(self) -> dict:
        """
        :return: The node as dictionary ( node_id, node_data )
        """
        string_pos = str(self._pos)
        n_dict = {"pos": string_pos,
                  "id": self.get_id()}  # representing as the json file.
        return n_dict

    def all_edges_out_dict(self) -> dict:
        """
        :return: all edges from the node as dictionary
        """
        return self.edges_out

    def all_edges_in_dict(self) -> dict:
        """
        :return: all edges to the node as dictionary
        """
        return self.edges_in

    def __eq__(self, other):
        """
        compares two nodes only on id and position.
        :param other: Node
        :return: True if they are equal, false if are not.
        """
        if isinstance(other, Node):
            if self._id == other._id and self._pos == other._pos:
                return True

        return False

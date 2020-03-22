from typing import Set

from ..edge import Edge
from ..node import Node
from ..exceptions import NodeNotInGraphError


class GraphLike:
    """Base object that specifies interface of all Graph objects.

    Graphs holds references to precreated Nodes and Edges and implements basic properties of node calculation in scope
    of graph object. Nodes and Edges are referenced by objects."""

    nodes: Set[Node]
    edge: Set[Edge]

    def __init__(self):
        """Start with empty sets of nodes and edges."""
        self.nodes = set()
        self.edges = set()

    def edges_from(self, node: Node) -> Set[Edge]:
        """Get edges outgoing from node.

        :param node: node from which edge objects goes out.
        :return: edge objects that exists both in nodes edges out set and graphs edges.
        :raises: NodeNotInGraphError.
        """
        if node not in self.nodes:
            raise NodeNotInGraphError(node)
        return self.edges.intersection(node.edges_from)

    def edges_to(self, node: Node) -> Set[Edge]:
        """Get edges ingoing to node.

        :param node: node to which edge objects goes in.
        :return: edge objects that exists both in nodes edges to set and graphs edges.
        :raises: NodeNotInGraphError.
        """
        if node not in self.nodes:
            raise NodeNotInGraphError(node)
        return self.edges.intersection(node.edges_to)

    def degree_out(self, node: Node) -> int:
        """Calculate out degree of node.

        :param node: node object for which degree out should be calculated.
        :return: out degree of node in scope of graph.
        :raises: NodeNotInGraphError.
        """
        return len(self.edges_from(node))

    def degree_in(self, node: Node) -> int:
        """Calculate in degree of node.

        :param node: node object for which degree in should be calculated.
        :return: in degree of node in scope of graph.

        """
        return len(self.edges_to(node))

    def add_node(self, node: Node):
        """Add node reference to graph object.

        :param node: node to add.
        """
        self.nodes.add(node)

    def add_edge(self, edge: Edge):
        """Add edge reference to graph. Raises error if either of nodes of edge is not in graph.

        :param edge: edge to add.
        :raises: NodeNotInGraphError.
        """
        if edge.node_to not in self.nodes:
            raise NodeNotInGraphError(edge.node_to, edge)
        elif edge.node_from not in self.nodes:
            raise NodeNotInGraphError(edge.node_from, edge)
        self.edges.add(edge)

    def find_similar_nodes(self, node: Node) -> Set[Node]:
        """Get nodes of same type as input node. Input node does not have to be in graph.

        :param node: reference node.
        :return: set of nodes in graph.
        """
        return {g_node for g_node in self.nodes if g_node == node}

    def find_similar_edges(self, edge: Edge) -> Set[Edge]:
        """Get edges of same type as input edge. Input edge needs not to be in graph.

        :param edge: reference edge.
        :return: set of edges in graph.
        """
        return {g_edge for g_edge in self.edges if g_edge == edge}
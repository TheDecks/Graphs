from typing import Set

from .bases.graph_like import GraphLike, Node, Edge
from .mask import Mask


class Universe(GraphLike):
    """Graph structure that used to represent supergraph of any relation."""

    def __init__(self, nodes: Set[Node], edges: Set[Edge]):
        """Initialize relation as a set of nodes and edges. Edges are added one-by-one as to assure the integrity
         of a structure.

         root_graphs is a subgraph of universe that does not have "dead ends"

        :param nodes: node objects.
        :param edges: edge objects.
        """
        super().__init__()
        self.nodes = nodes
        for edge in edges:
            self.add_edge(edge)
        self.leaves = {node for node in self.nodes if self.degree_out(node) == 0}
        root_nodes = self.nodes - self.leaves
        root_edges = {edge for edge in self.edges if edge.node_to not in self.leaves}
        self.root_graph = Mask(root_nodes, root_edges, self)


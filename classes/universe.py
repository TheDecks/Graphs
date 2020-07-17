from typing import Set, List

from .bases.graph_like import GraphLike, Node, Edge
from .mask import Mask
import random


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

    def re_root(self, omit_edge_label: List[str]):
        """Redefine the root graph for universe. It omits edges whose labels are in omit_edge_labels and also does not
        store references for nodes they point at. This is used mostly to get rid of uniquely identifying nodes.

        :param omit_edge_label: list of edge labels not to be included in root graph.
        """
        self.leaves = {node for node in self.nodes
                       if any([edge.label in omit_edge_label for edge in self.edges_to(node)])}
        root_nodes = self.nodes - self.leaves
        root_edges = {edge for edge in self.edges if edge.node_to in root_nodes and edge.node_from in root_nodes}
        self.root_graph = Mask(root_nodes, root_edges, self)

    def get_random_full_mask(self, size: int, breadth_first_preference: float = 0.5, from_root: bool = True) -> Mask:
        """Create a mask, that is a graphlike structure which can be thought of as a subgraph of universe. It holds only
        the references for known nodes and edges. This method create subgraph induced by its nodes.

        :param size: number of nodes to be included in the mask.
        :param breadth_first_preference: value from [0,0.5) promotes depth first like behaviour, value from (0.5, 1]
            promotes breadth first like
        :param from_root: whether to create the mask from root graph or whole universe.
        """
        if from_root:
            g = self.root_graph
        else:
            g = self
        nodes = set(random.sample(g.nodes, 1))
        while len(nodes) < size:
            to_extend = {edge.node_to for node in nodes
                         for edge in g.edges_from(node)
                         if random.random() < breadth_first_preference}
            to_extend.update({edge.node_from for node in nodes
                              for edge in g.edges_to(node)
                              if random.random() < breadth_first_preference})
            if len(to_extend) > size - len(nodes):
                to_extend = set(random.sample(to_extend, k=size - len(nodes)))
            nodes.update(to_extend)
        edges = {edge for edge in g.edges if edge.node_from in nodes and edge.node_to in nodes}
        return Mask(nodes, edges, g)

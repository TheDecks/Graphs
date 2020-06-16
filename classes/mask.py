from typing import Set, List, Tuple, Dict

from .bases.graph_like import GraphLike, Node, Edge


class Mask(GraphLike):
    """Graph object that restricts nodes and edges of another graph structure. Basically masks work like a way of
    specifying a subgraph."""

    def __init__(self, nodes: Set[Node], edges: Set[Edge], universe: GraphLike):
        """Create mask as a set of nodes and edges from graph-like structure.

        :param nodes: set of nodes to be included in subgraph.
        :param edges: set of edges to be included in subgraph.
        :param universe: reference to supergraph onto which a mask is built.
        """
        super().__init__()
        self.nodes = nodes
        self.edges = edges
        self.universe = universe

    def higher_dimension_masks(self, skip_edge_labels: Tuple[str] = ()) -> List['Mask']:
        """Get subgraphs that are grown by one edge onto the universe.

        :param skip_edge_labels: container with forbidden labels of edges.
        :return: list of subgraphs that has one more edge than current mask, built into the same universe.
        """

        out_edges = {
            edge for node in self.nodes
            for edge in self.universe.edges_from(node).union(self.universe.edges_to(node))
            if edge.label not in skip_edge_labels
        }
        # out_edges = {edge for node in self.nodes
        #              for edge in self.universe.edges_from(node) | self.universe.edges_to(node)
        #              if edge.label not in skip_edge_labels}
        possible_grows = out_edges - self.edges
        new_masks = []
        for edge in possible_grows:
            this_mask = Mask(self.nodes, self.edges, self.universe)
            if edge.node_to not in this_mask.nodes:
                this_mask.add_node(edge.node_to)
            if edge.node_from not in this_mask.nodes:
                this_mask.add_node(edge.node_from)
            this_mask.add_edge(edge)
            new_masks.append(this_mask)
        return new_masks

    # TODO: Create logic of mapping finding. Exact graph mapping
    def find_instances_in(self, graph: GraphLike) -> List[Tuple[Dict[Node, Node], Dict[Edge, Edge]]]: ...


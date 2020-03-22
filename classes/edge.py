from .bases.node_like import NodeLike
from .bases.edge_like import EdgeLike


class Edge(EdgeLike):
    """Actual Edge objects used in project."""

    node_from: NodeLike
    node_to: NodeLike

    def __init__(self, node_from: NodeLike, node_to: NodeLike, label: str):
        """Initialize object and add reference of edge to node objects. Hash is precalculated as it is expected to be
        referenced multiple times.

        :param node_from: starting point of edge.
        :param node_to: end point of edge.
        :param label: relation type between nodes.
        """
        super().__init__(node_from, node_to, label)
        self.hash = hash((self.node_from.id, self.node_to.id))
        self.node_from.edges_from.add(self)
        self.node_to.edges_to.add(self)

    def __hash__(self):
        """Hash value for sets and dictionaries."""
        return self.hash

    def __eq__(self, other: EdgeLike):
        """Edges are compared based on their starting point, relation and ending point."""
        return (self.node_from, self.label, self.node_to) == (other.node_from, other.label, other.node_to)

    def __str__(self):
        """Helper for printing purposes."""
        return f"<{type(self).__name__} {str(self.node_from)} ({self.label})-> {str(self.node_to)}>"

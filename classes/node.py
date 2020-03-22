from typing import Set

from .bases.node_like import NodeLike
from .bases.edge_like import EdgeLike


class Node(NodeLike):
    """Actual Node class, that acts as a container for a value."""

    edges_to: Set[EdgeLike]
    edges_from: Set[EdgeLike]

    def __str__(self):
        """Helper for printing purposes."""
        return f"<{type(self).__name__} {self.id}: {self.value}>"

from typing import Optional

from .bases.edge_like import EdgeLike
from .bases.node_like import NodeLike


class Error(Exception):
    """Base error class for inheritance in project, to be able to distinguish exceptions risen from this module."""

    def __init__(self):
        pass


class NodeNotInGraphError(Error):
    """Risen in case when specified Node is not in graph. Optional argument serves the purpose to also list
    Edge that tries to reference the node."""

    def __init__(self, node: NodeLike, edge: Optional[EdgeLike] = None):
        """
        :param node: referenced Node object.
        :param edge: referencing Edge object.
        """
        self.node = node
        self.edge = edge

    def __str__(self):
        return f"{str(self.node)}{f' of edge {str(self.edge)}' if self.edge is not None else ''} not in graph."

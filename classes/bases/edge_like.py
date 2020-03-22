class EdgeLike:
    """Base class for directed edge objects. For typing specification and inheritance."""

    def __init__(self, node_from, node_to, label: str):
        """Base assigning of attributes for edge.

        :param node_from: starting point of edge.
        :param node_to: end point of edge.
        :param label: edge type identifier. Specifies relation between nodes.
        """
        self.label = label
        self.node_from = node_from
        self.node_to = node_to

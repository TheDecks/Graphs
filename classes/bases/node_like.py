from typing import Union


class NodeLike:
    """Base class for node objects. Mainly for inheritance and typing specification."""

    def __init__(self, _id: int, value: Union[str, int]):
        """Assigning values for node objects. Each node starts with an empty set of in- and out- going edges.
        Hash is precalculated, as it is expected to be referenced multiple times.

        :param _id: unique identifier for node, assigned by graph object.
        :param value: underlying value (label) of node.
        """
        self.id = _id
        self.value = value
        self.edges_to = set()
        self.edges_from = set()
        self.hash = hash(self.id)

    def __hash__(self):
        """Hash value for dictionaries and sets."""
        return self.hash

    def __eq__(self, other: 'NodeLike'):
        """Nodes are compared by their value."""
        return self.value == other.value

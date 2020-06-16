from typing import List, Union

from .node import NodeLike
from .edge import Edge


class Walk:

    def __init__(self, edge_sequence: List[Edge]):
        self.edges = edge_sequence

    @property
    def is_open(self) -> bool:
        return self.edges[0].node_from.id != self.edges[-1].node_to.id

    @property
    def is_closed(self) -> bool:
        return not self.is_open

    @property
    def is_trial(self) -> bool:
        traversed = set()
        for edge in self.edges:
            if edge in traversed:
                return False
            else:
                traversed.add(edge)
        return True

    @property
    def is_path(self) -> bool:
        traversed = set()
        visited = set()
        for edge in self.edges:
            if edge in traversed or edge.node_from in visited:
                return False
            else:
                traversed.add(edge)
                visited.add(edge.node_from)
        return True

    @property
    def is_circuit(self) -> bool:
        return self.is_path and self.is_closed

    @property
    def vertices(self) -> List[NodeLike]:
        return [self.edges[0].node_from] + [edge.node_to for edge in self.edges] if self.edges else []

    def __contains__(self, item: Union[Edge, NodeLike]):
        return item.hash in [edge.hash for edge in self.edges] if isinstance(item, Edge) \
            else item.id in [vertex.id for vertex in self.vertices]

    def __len__(self):
        return len(self.edges)

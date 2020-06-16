from typing import List, Dict, Tuple

from classes.bases.graph_like import GraphLike
from .bases.node_like import NodeLike
from .walk import Walk


class PathCompletion:

    path_stats_forward: Dict[Tuple[str, ...], Dict[str, float]]
    path_stats_backward: Dict[Tuple[str, ...], Dict[str, float]]

    def __init__(self, graph: GraphLike, labelling: Dict[NodeLike, str]):
        self.graph = graph
        self.labelling = labelling
        self.id_to_node_mapping = {node.id: node for node in self.graph.nodes}
        self.path_stats_forward = {}
        self.path_stats_backward = {}

    def _generate_paths(self, from_node: NodeLike):
        if self.labelling[from_node] is None:
            return []
        else:
            paths_to_extend = [Walk([edge]) for edge in self.graph.edges_from(from_node)
                               if self.labelling[edge.node_to] is not None]
            paths = []
            while paths_to_extend:
                this_path = paths_to_extend.pop()
                paths.append(this_path)
                this_edges = this_path.edges
                last_vertex = this_path.vertices[-1]
                extensions = [edge for edge in self.graph.edges_from(last_vertex)
                              if self.labelling[edge.node_to] is not None and
                              edge.node_to not in this_path]
                for extension in extensions:
                    paths_to_extend.append(Walk(this_edges + [extension]))
            return paths

    def calculate_path_stats(self):
        nodes = set(self.graph.nodes)
        while nodes:
            node = nodes.pop()
            for path in self._generate_paths(node):
                pre_path = tuple(vertex.value for vertex in path.vertices[:-1])
                pro_path = tuple(vertex.value for vertex in path.vertices[1:])
                finish = path.vertices[-1].value
                start = path.vertices[0].value
                if pre_path not in self.path_stats_forward:
                    self.path_stats_forward[pre_path] = {}
                if finish not in self.path_stats_forward[pre_path]:
                    self.path_stats_forward[pre_path][finish] = 1
                else:
                    self.path_stats_forward[pre_path][finish] += 1

                if pro_path not in self.path_stats_backward:
                    self.path_stats_backward[pro_path] = {}
                if start not in self.path_stats_backward[pro_path]:
                    self.path_stats_backward[pro_path][start] = 1
                else:
                    self.path_stats_backward[pro_path][start] += 1

    def get_backward_paths(self, from_node: NodeLike) -> List[Tuple[str, ...]]:
        candidate_start_nodes = [edge.node_from for edge in self.graph.edges_to(from_node)
                                 if edge.node_from in self.labelling and self.labelling[edge.node_from] is not None]
        paths_to_extend = [Walk([edge]) for node in candidate_start_nodes
                           for edge in self.graph.edges_to(node)
                           if self.labelling[edge.node_from] is not None]
        paths = [(node.value,) for node in candidate_start_nodes]
        while paths_to_extend:
            this_path = paths_to_extend.pop()
            paths.append(tuple(vertex.value for vertex in this_path.vertices))
            this_edges = this_path.edges
            first_vertex = this_path.vertices[0]
            extensions = [edge for edge in self.graph.edges_to(first_vertex)
                          if self.labelling[edge.node_from] is not None and
                          edge.node_from not in this_path]
            for extension in extensions:
                paths_to_extend.append(Walk([extension] + this_edges))
        return paths

    def get_forward_paths(self, from_node: NodeLike) -> List[Tuple[str, ...]]:
        candidate_start_nodes = [edge.node_to for edge in self.graph.edges_from(from_node)
                                 if edge.node_to in self.labelling and self.labelling[edge.node_to] is not None]
        paths_to_extend = [Walk([edge]) for node in candidate_start_nodes
                           for edge in self.graph.edges_from(node)
                           if edge.node_to in self.labelling and self.labelling[edge.node_to] is not None]
        paths = [(node.value,) for node in candidate_start_nodes]
        while paths_to_extend:
            this_path = paths_to_extend.pop()
            paths.append(tuple(vertex.value for vertex in this_path.vertices))
            this_edges = this_path.edges
            last_vertex = this_path.vertices[-1]
            extensions = [edge for edge in self.graph.edges_from(last_vertex)
                          if edge.node_to in self.labelling and self.labelling[edge.node_to] is not None and
                          edge.node_to not in this_path]
            for extension in extensions:
                paths_to_extend.append(Walk(this_edges + [extension]))
        return paths

    def update_stats(self, paths: List[Tuple[str, ...]], chosen_label: str, direction: str = "forward"):
        if direction == "forward":
            path_stats = self.path_stats_forward
        else:
            path_stats = self.path_stats_backward
        for path in paths:
            if path not in path_stats:
                path_stats[path] = {chosen_label: 1}
            else:
                if chosen_label not in path_stats[path]:
                    path_stats[path][chosen_label] = 1
                else:
                    path_stats[path][chosen_label] += 1

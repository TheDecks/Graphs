from typing import List, Dict, Set, Tuple

from classes.bases.graph_like import GraphLike
from .bases.node_like import NodeLike
import pandas as pd


class NodeFeatureExtractor:

    current_neighbourhood_stats_from: Dict[NodeLike, Dict[str, int]]
    current_neighbourhood_stats_to: Dict[NodeLike, Dict[str, int]]

    def __init__(self, graph: GraphLike, labelling: Dict[NodeLike, str]):
        self.graph = graph
        self.labelling = labelling
        self.possible_labels = {label for label in self.labelling.values() if label is not None}
        self.representation = pd.DataFrame({'NodeId': [node.id for node in self.graph.nodes]})
        self.id_to_node_mapping = {node.id: node for node in self.graph.nodes}
        self.current_neighbourhood_stats_from = {}
        self.current_neighbourhood_stats_to = {}

    def report_degrees(self):
        self.representation["DegreeIn"] = self.representation.apply(
            lambda row: self.graph.degree_in(self.id_to_node_mapping[row["NodeId"]]),
            axis=1
        )
        self.representation["DegreeOut"] = self.representation.apply(
            lambda row: self.graph.degree_out(self.id_to_node_mapping[row["NodeId"]]),
            axis=1
        )

    def update_neighbourhood_stats(self):
        for node in self.graph.nodes:
            nbhd_labelling_from = [
                self.labelling[edge.node_to]
                for edge in self.graph.edges_from(node)
                if edge.node_to in self.labelling and self.labelling[edge.node_to] is not None
            ]
            nbhd_labelling_to = [
                self.labelling[edge.node_from]
                for edge in self.graph.edges_to(node)
                if edge.node_from in self.labelling and self.labelling[edge.node_from] is not None
            ]
            self.current_neighbourhood_stats_from[node] = NodeFeatureExtractor._aggregate_list(nbhd_labelling_from)
            self.current_neighbourhood_stats_to[node] = NodeFeatureExtractor._aggregate_list(nbhd_labelling_to)

    def calculate_label_proximity(self):
        self.update_neighbourhood_stats()
        for label in self.possible_labels:
            self.representation[label + "From"] = self.representation.apply(
                lambda row: self.current_neighbourhood_stats_from[
                    self.id_to_node_mapping[row["NodeId"]]
                ][label] if label in self.current_neighbourhood_stats_from[
                    self.id_to_node_mapping[row["NodeId"]]
                ] else 0,
                axis=1
            )
            self.representation[label + "To"] = self.representation.apply(
                lambda row: self.current_neighbourhood_stats_to[
                    self.id_to_node_mapping[row["NodeId"]]
                ][label] if label in self.current_neighbourhood_stats_to[
                    self.id_to_node_mapping[row["NodeId"]]
                ] else 0,
                axis=1
            )

    def get_representation_labels(self, nodes: Set[NodeLike]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        ids = {node.id for node in nodes}
        return (self.representation[self.representation["NodeId"].isin(ids)],
                self.representation[self.representation["NodeId"].isin(ids)]["NodeId"].apply(
                    lambda _id: self.labelling[self.id_to_node_mapping[_id]]
                    if self.id_to_node_mapping[_id] in self.labelling
                    else None
                ))

    def update_labelling(self, nodes_ids: List[int], labels: List[str]):
        self.labelling.update({
            self.id_to_node_mapping[_id]: label for _id, label in zip(nodes_ids, labels)
        })

    @staticmethod
    def _aggregate_list(to_aggregate: List[str]) -> Dict[str, int]:
        stats = {}
        for item in to_aggregate:
            if item not in stats:
                stats[item] = 1
            else:
                stats[item] += 1
        return stats

# as of https://www.genome.jp/kegg/xml/docs/

from typing import Tuple, Set, List, Dict

import requests
from xml.etree import ElementTree
from classes.universe import Universe, Edge, Node


class Pathway(Universe):
    """Represent pathway as universe graph. See https://www.genome.jp/kegg/xml/docs/ as reference for objectification"""

    entry_id_to_node_map: Dict[int, Node]

    @staticmethod
    def find_unreferenced(nodes: Set[Node]) -> Tuple[Set[Node], Set[Edge]]:
        """Get unreferenced edges and nodes as well as trees they are spanning.

        :param nodes: set of edges from which unreferenced components should be found.
        :return: edges and nodes that are disconnected from the rest.
        """
        unref_nodes = set()
        unref_edges = set()
        for node in nodes:
            if not node.edges_to:
                unref_nodes.add(node)
                unref_nodes.update({edge.node_to for edge in node.edges_from})
                unref_edges |= node.edges_from
        return unref_nodes, unref_edges

    def __init__(
            self,
            entries: List[ElementTree.Element],
            relations: List[ElementTree.Element],
            reactions: List[ElementTree.Element]
    ):
        self.no_nodes = 0
        self.entry_id_to_node_map = {}
        entry_nodes, entry_edges = self.parse_entry_xmls(entries)
        relation_nodes, relation_edges = self.parse_relation_xmls(relations)
        reaction_nodes, reaction_edges = self.parse_reaction_xmls(reactions)
        unreferenced_nodes, unreferenced_edges = Pathway.find_unreferenced(entry_nodes)
        entry_nodes -= unreferenced_nodes
        entry_edges -= unreferenced_edges

        super().__init__(
            nodes=entry_nodes | relation_nodes | reaction_nodes,
            edges=entry_edges | relation_edges | reaction_edges
        )

    @classmethod
    def from_url(cls, url: str):
        resp = requests.get(url)
        return cls.from_kgml(resp.text)

    @classmethod
    def from_kgml(cls, kgml_text: str):
        path_xml = ElementTree.fromstring(kgml_text)
        entries = []
        relations = []
        reactions = []
        for child in path_xml:
            if child.tag == 'entry':
                entries.append(child)
            elif child.tag == 'relation':
                relations.append(child)
            elif child.tag == 'reaction':
                reactions.append(child)
        return cls(entries, relations, reactions)

    def parse_entry_xmls(self, entries: List[ElementTree.Element]) -> Tuple[Set[Node], Set[Edge]]:
        nodes = set()
        edges = set()
        for entry in entries:
            n, e = self._parse_entry(entry)
            nodes.update(n)
            edges.update(e)
            self.no_nodes += len(n)
        return nodes, edges

    def _parse_entry(self, entry: ElementTree.Element) -> Tuple[Set[Node], Set[Edge]]:
        _id = self.no_nodes
        kegg_id = int(entry.get('id'))
        name = entry.get('name')
        _type = entry.get('type')
        # print(_id, kegg_id, name, _type)
        entry_node = Node(_id, 'entry')
        name_node = Node(_id + 1, name)
        type_node = Node(_id + 2, _type)
        name_edge = Edge(entry_node, name_node, 'name')
        type_edge = Edge(entry_node, type_node, 'type')
        self.entry_id_to_node_map[kegg_id] = entry_node
        return {entry_node, name_node, type_node}, {name_edge, type_edge}

    def parse_relation_xmls(self, relations: List[ElementTree.Element]) -> Tuple[Set[Node], Set[Edge]]:
        nodes = set()
        edges = set()
        for relation in relations:
            n, e = self._parse_relation(relation)
            nodes.update(n)
            edges.update(e)
            self.no_nodes += len(n)
        return nodes, edges

    def _parse_relation(self, relation: ElementTree.Element) -> Tuple[Set[Node], Set[Edge]]:
        _id = self.no_nodes
        _type = relation.get('type')
        entry_1_id = int(relation.get('entry1'))
        entry_2_id = int(relation.get('entry2'))
        relation_node = Node(_id, 'relation')
        type_node = Node(_id + 1, _type)
        entry_to_relation_node = self.entry_id_to_node_map[entry_1_id]
        relation_to_entry_node = self.entry_id_to_node_map[entry_2_id]
        entry_to_relation_edge = Edge(relation_node, entry_to_relation_node, 'entry_to_relation')
        relation_to_entry_edge = Edge(relation_node, relation_to_entry_node, 'relation_to_entry')
        type_edge = Edge(relation_node, type_node, 'type')
        subtype_nodes, subtype_edges = self._parse_subtypes(relation_node, relation.findall('subtype'))
        return {relation_node, type_node} | subtype_nodes,\
               {entry_to_relation_edge, relation_to_entry_edge, type_edge} | subtype_edges

    def _parse_subtypes(self, relation_node: Node, subtypes: List[ElementTree.Element]) -> Tuple[Set[Node], Set[Edge]]:
        _id = self.no_nodes + 2
        nodes = set()
        edges = set()
        for subtype in subtypes:
            subtype_name = subtype.get('name')
            subtype_value = subtype.get('value')
            subtype_node = Node(_id, subtype_name)
            nodes.add(subtype_node)
            _id += 1
            subtype_edge = Edge(relation_node, subtype_node, 'subtype')
            if subtype_name == 'compound':
                value_entry_node = self.entry_id_to_node_map[int(subtype_value)]
                value_edge = Edge(subtype_node, value_entry_node, 'value')
            else:
                value_node = Node(_id, subtype_value)
                value_edge = Edge(subtype_node, value_node, 'value')
                nodes.add(value_node)
            edges.update({subtype_edge, value_edge})
        return nodes, edges

    def parse_reaction_xmls(self, reactions: List[ElementTree.Element]) -> Tuple[Set[Node], Set[Edge]]:
        nodes = set()
        edges = set()
        for reaction in reactions:
            n, e = self._parse_reaction(reaction)
            nodes.update(n)
            edges.update(e)
            self.no_nodes += len(n)
        return nodes, edges

    def _parse_reaction(self, reaction: ElementTree.Element) -> Tuple[Set[Node], Set[Edge]]:
        _id = self.no_nodes
        _type = reaction.get('type')
        name = reaction.get('name')
        entry_to_reaction_id = int(reaction.get('id'))
        reaction_node = Node(_id, 'reaction')
        type_node = Node(_id + 1, _type)
        name_node = Node(_id + 2, name)
        entry_to_reaction_node = self.entry_id_to_node_map[entry_to_reaction_id]
        entry_to_reaction_edge = Edge(reaction_node, entry_to_reaction_node, 'entry_to_reaction')
        type_edge = Edge(reaction_node, type_node, 'type')
        name_edge = Edge(reaction_node, name_node, 'name')
        product_substrate_edges = self._parse_products_substrates(
            reaction_node,
            reaction.findall('product') + reaction.findall('substrate')
        )
        return {reaction_node, type_node, name_node}, \
               {entry_to_reaction_edge, type_edge, name_edge} | product_substrate_edges

    def _parse_products_substrates(
            self, reaction_node: Node, products_substrates: List[ElementTree.Element]
    ) -> Set[Edge]:
        edges = set()
        for item in products_substrates:
            item_node = self.entry_id_to_node_map[int(item.get('id'))]
            edge_label = 'reaction_to_product' if item.tag == 'product' else 'substrate_to_reaction'
            edges.add(Edge(reaction_node, item_node, edge_label))
        return edges


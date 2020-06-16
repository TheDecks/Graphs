from classes.node import Node
from classes.edge import Edge
from classes.bases.graph_like import GraphLike
from classes.node_feature_extractor import NodeFeatureExtractor
from classes.graph_vectorizer import GraphVectorizer
from classes.path_completion import PathCompletion
import pprint as pp

n1 = Node(1, "A")
n2 = Node(2, "B")
n3 = Node(3, "C")
n4 = Node(4, "D")
n5 = Node(5, "B")
nodes = [n1, n2, n3, n4, n5]

e1 = Edge(n1, n2, "A->B")
e2 = Edge(n2, n3, "B->C")
e3 = Edge(n3, n1, "C->A")
e4 = Edge(n3, n4, "C->D")
e5 = Edge(n5, n3, "B->C")
e6 = Edge(n5, n4, "B->D")
e7 = Edge(n1, n5, "A->B")
edges = [e1, e2, e3, e4, e5, e6, e7]

g = GraphLike()
for node in nodes:
    print(node)
    g.add_node(node)

for edge in edges:
    print(edge)
    g.add_edge(edge)

for node in g.nodes:
    print(node)
    print(g.edges_to(node))
    print(g.edges_from(node))

fe = NodeFeatureExtractor(g, labelling={node: node.value for node in nodes})
fe.update_neighbourhood_stats()
fe.report_degrees()
fe.calculate_label_proximity()
print(fe.representation)

gv = GraphVectorizer(g)
gv.set_up_neural_network(3)
gv.adjust_network(single_learn_repetition=10)
n2v = gv.node_to_vector()
print(n2v)

for node in nodes:
    print(node)
    g.add_node(node)

for edge in edges:
    print(edge)
    g.add_edge(edge)

for node in g.nodes:
    print(node)
    print(g.edges_to(node))
    print(g.edges_from(node))

pc = PathCompletion(g, labelling={node: node.value if node.id != 4 else None for node in nodes})
print(pc.graph.edges_to(n4))

pc.calculate_path_stats()
print(pc.graph.edges_to(n4))

pp.pprint(pc.path_stats)

paths = pc.get_backward_paths(n4)

for path in paths:
    print('->'.join(str(vertex.value) for vertex in path.vertices))

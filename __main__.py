from kegg.pathway import Pathway
from classes.graph_vectorizer import GraphVectorizer
from classes.node_feature_extractor import NodeFeatureExtractor
from classes.helpers.neural_network import *


# Create Pathway object as graph
hsa00010_pathway = Pathway.from_url("http://rest.kegg.jp/get/hsa00010/kgml")

# show information about nodes and edges in pathway
root = hsa00010_pathway.root_graph
for node in root.nodes:
    edges_from = root.edges_from(node)
    if edges_from:
        print(node)
        print('\n\t'.join(str(edge) for edge in edges_from))

# Create a vectorizing object
gv = GraphVectorizer(hsa00010_pathway)
gv.set_up_neural_network()
gv.adjust_network()

structure_vector_representation = gv.node_to_vector()

# Create node extracting object. Here it acts as it knows all labels
nfe = NodeFeatureExtractor(root, {node: node.value for node in root.nodes})
nfe.report_degrees()
nfe.calculate_label_proximity()

aggregate_node_statistics = nfe.representation

# Neural network playground
test_set = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [0, 0, 0]
])
expected_result = np.array([[1, 0, 1, 0], [0, 1, 0, 1]]).T
network = NeuralNetwork(test_set, expected_result, 0.1, mse)
network.add_layer(4, sigmoid)
network.construct_output_layer(sigmoid)
no_iter = 1500
for _ in range(no_iter):
    network.feed_forward()
    network.back_propagate()

print(network.output)

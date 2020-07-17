from typing import Optional

from classes.bases.graph_like import GraphLike, Walk, Node
from classes.helpers.neural_network import *
import numpy as np


class SPKernel:

    def __init__(self, g_1: GraphLike, g_2: GraphLike):
        self.g_1 = g_1
        self.g_2 = g_2
        self.triplets = []
        self.co_occurring = []
        self.g_1_samples = []
        self.g_2_samples = []
        self.neural_network = None

    def _generate_paths(self, from_node: Node, graph: int = 1):
        if graph == 1:
            graph = self.g_1
        else:
            graph = self.g_2
        paths_to_extend = [Walk([edge]) for edge in graph.edges_from(from_node)]
        paths = []
        while paths_to_extend:
            this_path = paths_to_extend.pop()
            this_edges = this_path.edges
            last_vertex = this_path.vertices[-1]
            extensions = [edge for edge in graph.edges_from(last_vertex) if
                          edge.node_to not in this_path]
            for extension in extensions:
                paths_to_extend.append(Walk(this_edges + [extension]))
            paths.append(this_path)
        return paths

    def populate_samples(self):
        for node in self.g_1.nodes:
            paths = self._generate_paths(node, 1)
            for path in paths:
                vertices = path.vertices
                main_sample = (vertices[0].value, vertices[-1].value, len(vertices)-1)
                if main_sample not in self.triplets:
                    self.triplets.append(main_sample)
                main_ind = self.triplets.index(main_sample)
                samples = [(vertices[0].value, vertices[i].value, i) for i in range(1, len(vertices) - 2)]
                for sample in samples:
                    if sample not in self.triplets:
                        self.triplets.append(sample)
                    sample_ind = self.triplets.index(sample)
                    self.co_occurring.append((main_ind, sample_ind))
                self.g_1_samples.extend(samples)
                self.g_1_samples.extend(main_sample)
        for node in self.g_2.nodes:
            paths = self._generate_paths(node, 2)
            for path in paths:
                vertices = path.vertices
                main_sample = (vertices[0].value, vertices[-1].value, len(vertices)-1)
                if main_sample not in self.triplets:
                    self.triplets.append(main_sample)
                main_ind = self.triplets.index(main_sample)
                samples = [(vertices[0].value, vertices[i].value, i) for i in range(1, len(vertices) - 2)]
                for sample in samples:
                    if sample not in self.triplets:
                        self.triplets.append(sample)
                    sample_ind = self.triplets.index(sample)
                    self.co_occurring.append((main_ind, sample_ind))
                self.g_2_samples.extend(samples)
                self.g_2_samples.extend(main_sample)

    @property
    def q_g_1(self):
        return np.transpose(np.array([[self.g_1_samples.count(triplet) for triplet in self.triplets]]))

    @property
    def q_g_2(self):
        return np.transpose(np.array([[self.g_2_samples.count(triplet) for triplet in self.triplets]]))

    def set_up_neural_network(self,
                              target_feature_number: Optional[int] = None,
                              learning_rate: float = 0.0000001,
                              hidden_layer_activation_function: ActivationFunction = tanh,
                              output_layer_activation_function: ActivationFunction = softmax,
                              cost_function: CostFunction = mse
                              ):
        N = len(self.triplets)
        target_feature_number = target_feature_number if target_feature_number else int(0.1 * N)
        self.neural_network = NeuralNetwork(
            np.zeros((1, N)),
            np.zeros((1, N)),
            learning_rate=learning_rate,
            cost_function=cost_function
        )
        self.neural_network.add_layer(target_feature_number, activation_function=hidden_layer_activation_function)
        self.neural_network.construct_output_layer(output_layer_activation_function)

    def adjust_network(self, single_learn_repetition: int = 1):
        N = len(self.triplets)
        for word, context in self.co_occurring:
            one_hot_input = np.zeros((1, N))
            one_hot_input[(0, word)] = 1
            one_hot_output = np.zeros((1, N))
            one_hot_output[(0, context)] = 1
            self.neural_network.update_input_output(one_hot_input, one_hot_output)
            for _ in range(single_learn_repetition):
                self.neural_network.loop()

    @property
    def M(self):
        return np.diag(np.sum(self.neural_network.hidden_layers[0].weights**2, axis=1))

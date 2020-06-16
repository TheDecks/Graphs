from typing import Optional

from classes.bases.graph_like import GraphLike, Walk
from classes.helpers.neural_network import *
import numpy as np


class RWKernel:

    def __init__(self, g_1: GraphLike, g_2: GraphLike):
        self.g_1 = g_1
        self.g_2 = g_2
        self.triplets = []
        self.co_occurring = []
        self.g_1_samples = []
        self.g_2_samples = []
        self.neural_network = None

    def populate_samples(self, no_walks: int, max_walk_length: int = 10):
        for _ in range(no_walks):
            w_1 = self.g_1.generate_random_walk(max_length=max_walk_length)
            w_2 = self.g_1.generate_random_walk(max_length=max_walk_length)
            self.g_1_samples.extend(self.generate_samples_from_walk(w_1))
            self.g_2_samples.extend(self.generate_samples_from_walk(w_2))

    def generate_samples_from_walk(self, walk: Walk):
        samples = []
        vertices = walk.vertices
        for start_node in range(0, len(vertices) - 1):
            this_node_created = []
            for length in reversed(range(1, len(vertices) - start_node)):
                sample = (vertices[start_node].value, vertices[start_node+length].value, length)
                samples.append(sample)
                if sample not in self.triplets:
                    self.triplets.append(sample)
                this_index = self.triplets.index(sample)
                for already_done in this_node_created:
                    self.co_occurring.append((already_done, this_index))
                this_node_created.append(this_index)
        return samples

    @property
    def q_g_1(self):
        return np.transpose(np.array([[self.g_1_samples.count(triplet) for triplet in self.triplets]]))

    @property
    def q_g_2(self):
        return np.transpose(np.array([[self.g_2_samples.count(triplet) for triplet in self.triplets]]))

    def set_up_neural_network(self,
                              target_feature_number: Optional[int] = None,
                              learning_rate: float = 0.00001,
                              hidden_layer_activation_function: ActivateFunction = tanh,
                              output_layer_activation_function: ActivateFunction = softmax,
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

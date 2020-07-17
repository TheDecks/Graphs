from typing import Optional

from classes.bases.graph_like import GraphLike
from classes.helpers.neural_network import *
import numpy as np


class WLKernel:

    def __init__(self, g_1: GraphLike, g_2: GraphLike):
        self.g_1 = g_1
        self.g_2 = g_2
        self.current_labels_g1 = {node: node.value for node in self.g_1.nodes}
        self.current_labels_g2 = {node: node.value for node in self.g_2.nodes}
        self.co_occurring = []
        self.labels = []
        self.g_1_counts = {}
        self.g_2_counts = {}
        self.neural_network = None
        self.labelling_count = 1

    def loop(self, times: int = 5):
        for _ in range(times):
            labelling = self.relabel()
            self.populate_co_occurrences(labelling)
        self.relabel()

    def relabel(self):
        relabelling = {}
        future_g1_labels = {}
        for node in self.g_1.nodes:
            lab = self.current_labels_g1[node]
            if lab not in self.labels:
                self.labels.append(lab)
            if lab not in self.g_1_counts:
                self.g_1_counts[lab] = 1
            else:
                self.g_1_counts[lab] += 1
            new_lab = (lab, tuple(sorted([self.current_labels_g1[edge.node_to] for edge in self.g_1.edges_from(node)])))
            if new_lab not in relabelling:
                relabelling[new_lab] = self.labelling_count
                self.labelling_count += 1
            future_g1_labels[node] = relabelling[new_lab]

        future_g2_labels = {}
        for node in self.g_2.nodes:
            lab = self.current_labels_g2[node]
            if lab not in self.labels:
                self.labels.append(lab)
            if lab not in self.g_2_counts:
                self.g_2_counts[lab] = 1
            else:
                self.g_2_counts[lab] += 1
            new_lab = (lab, tuple(sorted([self.current_labels_g2[edge.node_to] for edge in self.g_2.edges_from(node)])))
            if new_lab not in relabelling:
                relabelling[new_lab] = self.labelling_count
                self.labelling_count += 1
            future_g2_labels[node] = relabelling[new_lab]
        self.current_labels_g1 = future_g1_labels
        self.current_labels_g2 = future_g2_labels

        return relabelling

    def populate_co_occurrences(self, relabelling):
        for tree_1, new_label_1 in relabelling.items():
            for tree_2, new_label_2 in relabelling.items():
                if new_label_1 == new_label_2:
                    continue
                if WLKernel.are_co_occurring(tree_1, tree_2):
                    self.co_occurring.append((new_label_1, new_label_2))

    @staticmethod
    def are_co_occurring(tree_1, tree_2):
        if tree_1[0] == tree_2[0]:
            if len(tree_1[1]) >= len(tree_2[1]):
                supertree = list(tree_1[1])
                subtree = list(tree_2[1])
            else:
                supertree = list(tree_2[1])
                subtree = list(tree_1[1])
            if len(supertree) - len(subtree) > 1:
                return False
            mismatch = 0
            while supertree:
                try:
                    subtree.remove(supertree.pop())
                except ValueError:
                    mismatch += 1
                    if mismatch > 1:
                        return False
        else:
            return tree_1[1] == tree_2[1]

    @property
    def q_g_1(self):
        return np.transpose(np.array([[
            self.g_1_counts[label] if label in self.g_1_counts else 0 for label in self.labels
        ]]))

    @property
    def q_g_2(self):
        return np.transpose(np.array([[
            self.g_2_counts[label] if label in self.g_2_counts else 0 for label in self.labels
        ]]))

    def set_up_neural_network(self,
                              target_feature_number: Optional[int] = None,
                              learning_rate: float = 0.0001,
                              hidden_layer_activation_function: ActivationFunction = tanh,
                              output_layer_activation_function: ActivationFunction = softmax,
                              cost_function: CostFunction = mse
                              ):
        N = len(self.labels)
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
        N = len(self.labels)
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

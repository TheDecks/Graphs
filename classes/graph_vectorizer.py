from typing import Optional, Generator, Tuple

from .bases.graph_like import GraphLike
from .node import Node
from .helpers.neural_network import *
import pandas as pd


class GraphVectorizer:
    """Class implementing a simple node2vec."""

    neural_network: Optional[NeuralNetwork]

    def __init__(self, graph: GraphLike):
        """Create object that can represent a graph by creating a numerical vector for each node.

        :param graph: graph to vectorize.
        """
        self.graph = graph
        self.nodes_ordered = sorted(self.graph.nodes, key=lambda node: node.id)
        self.N = len(self.graph.nodes)
        self.neural_network = None

    def set_up_neural_network(self,
                              target_feature_number: Optional[int] = None,
                              learning_rate: float = 0.1,
                              hidden_layer_activation_function: ActivationFunction = linear,
                              output_layer_activation_function: ActivationFunction = softmax,
                              cost_function: CostFunction = mse
                              ):
        """Node2Vec thrives on the idea similar to Word2Vec. It uses an arbitrary task for a neural network with single
        hidden layer to learn the weights in it and takes them as the representation of vertices.

        :param target_feature_number: how long should be the vector that each node will be represented by.
        :param learning_rate: learning rate of the network.
        :param hidden_layer_activation_function: activation function on the hidden layer.
        :param output_layer_activation_function: activation function on the output layer.
        :param cost_function: cost function to calculate the mismatch between expected output and NN output.
        """
        target_feature_number = target_feature_number if target_feature_number else int(0.1 * self.N)
        self.neural_network = NeuralNetwork(
            np.zeros((1, self.N)),
            np.zeros((1, self.N)),
            learning_rate=learning_rate,
            cost_function=cost_function
        )
        self.neural_network.add_layer(target_feature_number, activation_function=hidden_layer_activation_function)
        self.neural_network.construct_output_layer(output_layer_activation_function)

    def adjust_network(self,
                       no_walks: Optional[int] = None,
                       max_walk_length: Optional[int] = None,
                       window_weights: tuple = (1 / 2, 1, 0, 1, 1 / 2),
                       close_proximity_parameter: float = 1,
                       single_learn_repetition: int = 1
                       ):
        """Context is generated from random walks which are treated as sentences are in word2vect.

        :param no_walks: number of walks to generate.
        :param max_walk_length: maximal length of the walk.
        :param window_weights: controls the shape, size and importance of samples generated from random walks. Use 0 to
            indicate the place of node a window spans from.
        :param close_proximity_parameter: parameter to control the behaviour of walk generation.
        :param single_learn_repetition: how many times should the network be adjusted based on single training sample.
        """
        no_walks = no_walks if no_walks else 4 * self.N
        max_walk_length = max_walk_length if max_walk_length else int(self.N ** (1 / 2))
        context = self._generate_context(no_walks, max_walk_length, window_weights, close_proximity_parameter)
        for node_in, node_out, out_value in context:
            for _ in range(single_learn_repetition):
                self._single_sample_adjust(node_in, node_out, out_value)

    def _generate_context(self,
                          no_walks: int,
                          max_length: int,
                          weights: tuple,
                          close_proximity_parameter: float = 1
                          ) -> Generator[Tuple[Node, Node, int], None, None]:
        """Helper method for adjust_network. See above for parameters description."""
        target_node_position = weights.index(0)
        for j in range(no_walks):
            walk = self.graph.generate_random_walk(max_length, close_proximity_param=close_proximity_parameter)
            vertices = walk.vertices
            for i in range(len(vertices)):
                if i < target_node_position:
                    this_weights = weights[target_node_position - i:]
                    this_vertices = vertices
                else:
                    this_weights = weights
                    this_vertices = vertices[i - target_node_position:]
                if len(this_weights) < len(this_vertices):
                    this_vertices = this_vertices[:len(this_weights)]
                else:
                    this_weights = this_weights[:len(this_weights)]
                for output_vertex, output_weight in zip(this_vertices, this_weights):
                    if output_weight != 0:
                        yield vertices[i], output_vertex, output_weight

    def _single_sample_adjust(self, node_in: Node, node_out: Node, out_value: float):
        """Helper function for adjust_network. Given a training sample that can be represented as (Node1, Node2, weight)
        the input is one-hot vector for Node1 and expected output is one-hot vector for Node2 multiplied by weight.

        :param node_in: node onto which the window spans.
        :param node_out: node in context of node_in.
        :param out_value: importance of node_out for node_in.
        """
        one_hot_input = np.zeros((1, self.N))
        one_hot_input[(0, self.nodes_ordered.index(node_in))] = 1
        weight_hot_output = np.zeros((1, self.N))
        weight_hot_output[(0, self.nodes_ordered.index(node_out))] = out_value
        self.neural_network.update_input_output(one_hot_input, weight_hot_output)
        self.neural_network.loop()

    def node_to_vector(self):
        """Get current representation of nodes."""
        representation = self.neural_network.hidden_layers[0].weights
        return pd.DataFrame(
            data=representation,
            index=[node.id for node in self.nodes_ordered],
            columns=[f"F{i}" for i in range(representation.shape[1])]
        )

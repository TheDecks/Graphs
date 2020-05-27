from typing import Optional, Generator, Tuple

from .bases.graph_like import GraphLike
from .node import Node
from .helpers.neural_network import *
import pandas as pd


class GraphVectorizer:

    neural_network: Optional[NeuralNetwork]

    def __init__(self, graph: GraphLike):
        self.graph = graph
        self.nodes_ordered = sorted(self.graph.nodes, key=lambda node: node.id)
        self.N = len(self.graph.nodes)
        self.neural_network = None

    def set_up_neural_network(self,
                              target_feature_number: Optional[int] = None,
                              learning_rate: float = 0.1,
                              hidden_layer_activation_function: ActivateFunction = linear,
                              output_layer_activation_function: ActivateFunction = sigmoid,
                              cost_function: CostFunction = mse
                              ):
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
                       close_proximity_parameter: float = 1
                       ):
        no_walks = no_walks if no_walks else 4 * self.N
        max_walk_length = max_walk_length if max_walk_length else int(self.N ** (1 / 2))
        context = self._generate_context(no_walks, max_walk_length, window_weights, close_proximity_parameter)
        for node_in, node_out, out_value in context:
            self._single_sample_adjust(node_in, node_out, out_value)

    def _generate_context(self,
                          no_walks: int,
                          max_length: int,
                          weights: tuple,
                          close_proximity_parameter: float = 1
                          ) -> Generator[Tuple[Node, Node, int], None, None]:
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
        one_hot_input = np.zeros((1, self.N))
        one_hot_input[(0, self.nodes_ordered.index(node_in))] = 1
        weight_hot_output = np.zeros((1, self.N))
        weight_hot_output[(0, self.nodes_ordered.index(node_out))] = out_value
        self.neural_network.update_input_output(one_hot_input, weight_hot_output)
        self.neural_network.loop()

    def node_to_vector(self):
        representation = self.neural_network.hidden_layers[0].weights
        return pd.DataFrame(
            data=representation,
            index=[node.id for node in self.nodes_ordered],
            columns=[f"F{i}" for i in range(representation.shape[1])]
        )

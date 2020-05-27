from typing import List, Union
from .layer import Layer
from .activation_function import *
from .cost_function import *
import numpy as np


class NeuralNetwork:

    hidden_layers: List[Layer]
    output_layer: Union[Layer, None]

    def __init__(self,
                 layer_zero: np.ndarray, objective: np.ndarray,
                 learning_rate: float = 1, cost_function: CostFunction = cel
                 ):
        self.layer_zero = layer_zero
        self.objective = objective
        self.hidden_layers = []
        self.current_features = layer_zero.shape[1]
        self.output_layer = None
        self.output = np.zeros(objective.shape)
        self.cost_function = cost_function
        self.learning_rate = learning_rate
        self.adjusting_counter = 0

    def add_layer(self, number_of_neurons: int, activation_function: ActivateFunction = linear):
        self.hidden_layers.append(Layer(self.current_features, number_of_neurons, activation_function))
        self.current_features = number_of_neurons

    def construct_output_layer(self, activation_function: ActivateFunction = sigmoid):
        self.output_layer = Layer(self.current_features, self.objective.shape[1], activation_function)

    def feed_forward(self):
        current_output = self.layer_zero
        for layer in self.hidden_layers:
            current_output = layer.get_output(current_output)
        self.output = self.output_layer.get_output(current_output)

    def back_propagate(self):
        derivation_chain = [self.cost_function.derivative(self.output, self.objective)]
        layers_to_traverse = self.hidden_layers + [self.output_layer]
        for layer in reversed(layers_to_traverse):
            derivation_chain.append(derivation_chain[-1] * layer.get_derivative_component())
            derivation_chain.append(np.dot(derivation_chain[-1], layer.weights.T))
        current_input = self.layer_zero
        for layer_level, layer in enumerate(layers_to_traverse):
            d_weight = np.dot(current_input.T, derivation_chain[-2 - 2 * layer_level])
            current_input = layer.last_output
            layer.weights += self.learning_rate * d_weight

    def loop(self):
        self.feed_forward()
        self.back_propagate()
        self.adjusting_counter += 1

    def adjust_weights(self, iterations: int):
        for _ in range(iterations):
            self.loop()

    def update_input_output(self, new_layer_zero: np.ndarray, new_objective: np.ndarray):
        self.layer_zero = new_layer_zero
        self.objective = new_objective

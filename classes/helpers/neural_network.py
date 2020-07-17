from typing import List, Union
from .layer import Layer
from .activation_function import *
from .cost_function import *
import numpy as np


class NeuralNetwork:
    """A naive implementation of neural network, that utilises feedfroward and backpropagation."""

    hidden_layers: List[Layer]
    output_layer: Union[Layer, None]

    # TODO: Reimplement the definition of neural network, to only take the sizes of input and output, to work on
    #  samples
    def __init__(self,
                 layer_zero: np.ndarray, objective: np.ndarray,
                 learning_rate: float = 1, cost_function: CostFunction = cel
                 ):
        """Initially the network was designed to work on gathered input with already known output. Hence there is
        layer_zero and objective parameters.

        :param layer_zero: Input to the network.
        :param objective: Expected output.
        :param learning_rate: Adjust the speed at which the neural network is learning.
        :param cost_function: CostFunction object that specifies how the error is calculated.
        """
        self.layer_zero = layer_zero
        self.objective = objective
        self.hidden_layers = []
        self.current_features = layer_zero.shape[1]
        self.output_layer = None
        self.output = np.zeros(objective.shape)
        self.cost_function = cost_function
        self.learning_rate = learning_rate
        self.adjusting_counter = 0

    # TODO: Add method that adds already created object, instead of creating one from parameters
    def add_layer(self, number_of_neurons: int, activation_function: ActivationFunction = linear):
        """Add next layer in the sequence.

        :param number_of_neurons: the number of neurons in new layer.
        :param activation_function: ActivationFunction object.
        """
        self.hidden_layers.append(Layer(self.current_features, number_of_neurons, activation_function))
        self.current_features = number_of_neurons

    def construct_output_layer(self, activation_function: ActivationFunction = sigmoid):
        """Create an output layer based on last layer neurons and expected output shape.

        :param activation_function: ActivationFunction on output layer.
        """
        self.output_layer = Layer(self.current_features, self.objective.shape[1], activation_function)

    def feed_forward(self):
        """Chain the outputs from input up to output layer."""
        current_output = self.layer_zero
        for layer in self.hidden_layers:
            current_output = layer.get_output(current_output)
        self.output = self.output_layer.get_output(current_output)

    def back_propagate(self):
        """Create a backwards chain of derivatives values and update the weights in layers."""
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
        """Single Epoch."""
        self.feed_forward()
        self.back_propagate()
        self.adjusting_counter += 1

    def adjust_weights(self, iterations: int):
        """Run for iterations epochs."""
        for _ in range(iterations):
            self.loop()

    def update_input_output(self, new_layer_zero: np.ndarray, new_objective: np.ndarray):
        """Method created to shift the implementation from all-known-at-start to one-training-sample at time.

        :param new_layer_zero: input for sample.
        :param new_objective: desired output for sample.
        """
        self.layer_zero = new_layer_zero
        self.objective = new_objective

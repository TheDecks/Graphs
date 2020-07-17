from typing import Union
import numpy as np
from .activation_function import ActivationFunction


class Layer:
    """A naive implementation of generic Layer in Neural Network."""

    last_output: Union[np.ndarray, None]

    # TODO: consider adding bias
    def __init__(self, previous_neurons: int, to_neurons: int, activation_function: ActivationFunction):
        """previous_neurons and to_neurons parameters specifies the shape of weights on given layer.

        :param previous_neurons: number of neurons in previous layer.
        :param to_neurons: desired number of neuron in this layer.
        :param activation_function: ActivationFunction object.
        """
        self.number_of_neurons = to_neurons
        self.weights = np.random.rand(previous_neurons, to_neurons) * 2 - 1
        self.activation_function = activation_function
        self.last_output = None

    def get_output(self, previous_layer_result: np.ndarray) -> np.ndarray:
        """Usage of this method allows for chaining the results from input layer up to output layer.

        :param previous_layer_result: Output from previous layer in the chain.
        :return: output of this layer.
        """
        self.last_output = self.activation_function(np.dot(previous_layer_result, self.weights))
        return self.last_output

    def get_derivative_component(self) -> np.ndarray:
        """Get the value of the derivative in this layer."""
        return self.activation_function.derivative(self.last_output)

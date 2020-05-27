from typing import Union
import numpy as np
from .activation_function import ActivateFunction


class Layer:

    last_output: Union[np.ndarray, None]

    # TODO: consider adding bias
    def __init__(self, previous_neurons: int, to_neurons: int, activation_function: ActivateFunction):
        self.number_of_neurons = to_neurons
        self.weights = np.random.rand(previous_neurons, to_neurons) * 2 - 1
        self.activation_function = activation_function
        self.last_output = None

    def get_output(self, previous_layer_result: np.ndarray):
        self.last_output = self.activation_function(np.dot(previous_layer_result, self.weights))
        return self.last_output

    def get_derivative_component(self):
        return self.activation_function.derivative(self.last_output)

from typing import Callable, Dict
import numpy as np


class ActivationFunction:
    """Generic class for activation functions in neural networks."""

    def __init__(self, function: Callable[[np.ndarray], np.ndarray], derivative: Callable[[np.ndarray], np.ndarray]):
        """The implementation is used for backpropagation method of adjusting neural networks, hence the interface of a
        function needs to specify the function and its derivative.

        :param function: Callable object that calculates the value of function.
        :param derivative: Callable object that calculates the value of derivative.
        """
        self.function = function
        self.derivative = derivative

    def __call__(self, _x: np.ndarray) -> np.ndarray:
        """For convenience, so that an instance can be directly called, omitting the referencing of function
        attribute."""
        return self.function(_x)


sigmoid = ActivationFunction(
    function=lambda x: 1/(1+np.exp(-x)),
    derivative=lambda x: x*(1-x)
)
relu = ActivationFunction(
    function=lambda x: np.where(x < 0, 0, x),
    derivative=lambda x: np.where(x > 0, 1, 0)
)
tanh = ActivationFunction(
    function=lambda x: np.tanh(x),
    derivative=lambda x: 1 - np.tanh(x)**2
)
linear = ActivationFunction(
    function=lambda x: x,
    derivative=lambda x: np.ones(x.shape)
)
softmax = ActivationFunction(
    function=lambda x: np.where(
        np.isnan(np.exp(x) / np.sum(np.exp(x), axis=1)[:, np.newaxis]),
        0, np.exp(x) / np.sum(np.exp(x), axis=1)[:, np.newaxis]
    ),
    derivative=lambda x: np.where(
        np.isnan(np.exp(x) / np.sum(np.exp(x), axis=1)[:, np.newaxis]),
        0, np.exp(x) / np.sum(np.exp(x), axis=1)[:, np.newaxis]
    ) - 1 / np.exp(x)
)

act_funs = {
    'sigmoid': sigmoid,
    'relu': relu,
    'tanh': tanh,
    'linear': linear,
    'softmax': softmax
}  # type: Dict[str, ActivationFunction]

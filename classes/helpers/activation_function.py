from typing import Callable, Dict
import numpy as np


class ActivateFunction:

    def __init__(self, function: Callable[[np.ndarray], np.ndarray], derivative: Callable[[np.ndarray], np.ndarray]):
        self.function = function
        self.derivative = derivative

    def __call__(self, _x: np.ndarray) -> np.ndarray:
        return self.function(_x)


sigmoid = ActivateFunction(
    function=lambda x: 1/(1+np.exp(-x)),
    derivative=lambda x: x*(1-x)
)
relu = ActivateFunction(
    function=lambda x: np.where(x < 0, 0, x),
    derivative=lambda x: np.where(x > 0, 1, 0)
)
tanh = ActivateFunction(
    function=lambda x: np.tanh(x),
    derivative=lambda x: 1 - np.tanh(x)**2
)
linear = ActivateFunction(
    function=lambda x: x,
    derivative=lambda x: np.ones(x.shape)
)

act_funs = {
    'sigmoid': sigmoid,
    'relu': relu,
    'tanh': tanh,
    'linear': linear
}  # type: Dict[str, ActivateFunction]

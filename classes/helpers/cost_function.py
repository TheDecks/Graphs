from typing import Callable, Dict
import numpy as np


class CostFunction:

    def __init__(self,
                 function: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 derivative: Callable[[np.ndarray, np.ndarray], np.ndarray]
                 ):
        self.function = function
        self.derivative = derivative

    def __call__(self, _y: np.ndarray, target: np.ndarray):
        return self.function(_y, target)


mse = CostFunction(
    function=lambda y, real_y: np.sum((real_y - y)**2),
    derivative=lambda y, real_y: 2 * (real_y - y)
)

cel = CostFunction(
    function=lambda y, real_y: -1 * (
            np.sum(real_y * np.log(y)) + np.sum((1-real_y)*np.log(1-y))
    ),
    derivative=lambda y, real_y: -(np.divide(real_y, y) - np.divide(1 - real_y, 1 - y))
)


cost_funcs = {
    'mse': mse,
    'cel': cel
}  # type: Dict[str, CostFunction]

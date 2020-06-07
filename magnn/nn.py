# coding=utf-8
"""
Here we make a Neural Network, a set of layers
"""
from typing import Sequence, Tuple, Iterator

from magnn.tensors import Tensor
from magnn.layer import Layer, Dropout


class Net:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers

    def forward(self, inputs: Tensor, **kwargs) -> Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs, **kwargs)
        return inputs

    def backward(self, grad: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def get_param_and_grads(self) -> Iterator[Tuple[Tensor, Tensor]]:
        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.grads[name]
                yield param, grad

    def refresh_dropouts(self):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.update_layer()

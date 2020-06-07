# coding=utf-8
"""
Optimizing parameters from gradients
ex: SGD,
"""
from magnn.nn import Net


class Optimizer:
    def step(self, net: Net) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr=0.005) -> None:
        self.lr = lr

    def step(self, net: Net) -> None:
        for param, grad in net.get_param_and_grads():
            param -= self.lr * grad  # go directly against the gradient

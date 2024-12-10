# coding=utf-8
"""
We'll have different layers here
Tanh
Linear
Conv??
"""
from typing import Dict, Callable

import numpy as np

from magnn.tensors import Tensor


# function definitions


def sigmoid(x: Tensor) -> Tensor:
    return np.power(1 + np.exp(-x), -1)


def sigmoid_p(x: Tensor) -> Tensor:
    return sigmoid(x) * sigmoid(-x)


def swish(x: Tensor) -> Tensor:
    return x * sigmoid(x)


def swish_p(x: Tensor) -> Tensor:
    return swish(x) + sigmoid(x) * (1 - swish(x))


def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)


def tanh_p(x: Tensor) -> Tensor:
    return 1 - (tanh(x) ** 2)


def relu(x: Tensor) -> Tensor:
    return np.max(x, 0)


def relu_p(x: Tensor) -> Tensor:
    return 1.0 * (x > 0)


def square(x: Tensor) -> Tensor:
    return x ** 2


def square_p(x: Tensor) -> Tensor:
    return 2 * x

def softmax(x: Tensor) -> Tensor:
    # num stability norm over each row in x (x are scores)
    # (i.e. across all key vectors for each query)
    exp_scores = np.exp(x - np.max(x, axis=-1, keepdims=True))  
    return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

def self_attn(x:Tensor, mask, W_KQV, W_out):
    K,Q,V = np.split(x@W_KQV, 3, axis=1)
    attn = softmax(K@Q.T / np.sqrt(X.shape[1])+mask)
    return attn@V@W_out, attn

class Layer:
    # ABC for layer, no impl here
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}
        self.inputs: Tensor = None
        pass

    def forward(self, inputs: Tensor, **kwargs) -> Tensor:
        """
        Produce outputs
        """
        raise NotImplementedError

    def backward(self, grad: Tensor, **kwargs) -> Tensor:
        """
        Backprop gradient
        """

        raise NotImplementedError


class Linear(Layer):
    """
    Linear regression
    """

    def __init__(self, input_size: int, output_size: int) -> None:
        # input will be (batch_size_ input_size)
        # output will be (batch_size_ output_size)
        super().__init__()

        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)

    def forward(self, inputs: Tensor, **kwargs) -> Tensor:
        """
        output = input @ w + b
        """
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad: Tensor, **kwargs) -> Tensor:
        """
        if y = f(x)
        where x = ab + c
        then dy/da = f' *b
        then dy/db = f' *a
        then dy/dc = f'
        """
        self.grads["b"] = np.sum(grad, axis=0)
        self.grads["w"] = self.inputs.T @ grad
        return grad @ self.params["w"].T


class Dropout(Layer):
    def __init__(self, size: int, prob: float = 0.5):
        super().__init__()
        self.prob: float = prob
        self.size: int = size
        self.r: Tensor = np.random.binomial(1, self.prob, size=self.size)

    def update_layer(self):
        self.r = np.random.binomial(1, self.prob, size=self.size)

    def forward(self, inputs: Tensor, training: bool = False) -> Tensor:
        if training:
            return self.r * inputs
        else:
            return (1 - self.prob) * inputs

    def backward(self, grad: Tensor, **kwargs) -> Tensor:
        return grad * self.r


F = Callable[[Tensor], Tensor]


class Activation(Layer):
    """
    Apply a function to inputs. Typically nonlinear.
    """

    def __init__(self, f: F, f_p: F) -> None:
        super().__init__()
        self.f = f
        self.f_p = f_p

    def forward(self, inputs: Tensor, **kwargs) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad: Tensor, **kwargs) -> Tensor:
        """
        now we're doing the chain rule here
        grad represents the current layer's gradient, and f_p is the gradient wrt the rest of the network
        :param **kwargs:
        """
        return grad * self.f_p(self.inputs)


class Tanh(Activation):
    def __init__(self):
        super().__init__(f=tanh, f_p=tanh_p)


class Sigmoid(Activation):
    def __init__(self):
        super().__init__(f=sigmoid, f_p=sigmoid_p)


class Swish(Activation):
    def __init__(self):
        super().__init__(f=swish, f_p=swish_p)


class ReLu(Activation):
    def __init__(self):
        super().__init__(f=relu, f_p=relu_p)


class Square(Activation):
    def __init__(self):
        super().__init__(f=square, f_p=square_p)

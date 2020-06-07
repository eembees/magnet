"""
Feeding things in batches
"""
from typing import NamedTuple, Iterator
from magnn.tensors import Tensor
import numpy as np


Batch = NamedTuple("Batch", [("inputs", Tensor), ("targets", Tensor)])


class DataIterator:
    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator[Batch]:
        raise NotImplementedError


class BatchIterator(DataIterator):
    def __init__(self, batch_size: int = 32, shuffle: bool = True):
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator[Batch]:
        starts = np.arange(0, len(inputs), self.batch_size)
        if self.shuffle:
            np.random.shuffle(starts)

        for start in starts:
            end = start + self.batch_size
            batch_in = inputs[start:end]
            batch_out = targets[start:end]
            yield Batch(batch_in, batch_out)


class OnlineIterator(BatchIterator):
    def __init__(self):
        super().__init__(batch_size=1, shuffle=True)


class Scaler:
    def __init__(self, scale_mean: bool = True, scale_var: bool = True) -> None:
        self.scale_mean = True
        self.scale_var = True
        self.means: Tensor = None
        self.vars: Tensor = None

    def fit(self, X: Tensor) -> None:
        self.means = np.mean(X, axis=0)
        self.vars = np.var(X, axis=0)

    def transform(self, X: Tensor) -> Tensor:
        _X = X.copy()
        if self.scale_mean:
            _X -= self.means
        if self.scale_var:
            _X /= self.vars
        return _X

# coding=utf-8
"""
We use this function to train our NN
"""
from typing import Union, List, Tuple
import numpy as np

from magnn.tensors import Tensor
from magnn.nn import Net
from magnn.loss import Loss, MSE
from magnn.optimize import Optimizer, SGD
from magnn.io import DataIterator, BatchIterator


def train(
    net: Net,
    inputs: Tensor,
    targets: Tensor,
    epochs: int = 5000,
    iterator: DataIterator = BatchIterator(),
    loss: Loss = MSE(),
    optim: Optimizer = SGD(),
    val_data: Union[Tuple[Tensor, Tensor], None] = None,
    val_frequency: int = 100,
    early_stopping: bool = False,
    early_stop_window: int = 100,
    early_stop_frequency: int = 100,
    verbose: bool = False,
) -> Tuple[List[float], List[float]]:
    losses = []
    val_losses = []
    for epoch in range(epochs):
        net.refresh_dropouts()

        epoch_loss = 0.0
        for i, batch in enumerate(iterator(inputs, targets)):
            predicted = net.forward(batch.inputs, training=True)
            epoch_loss += loss.loss(predicted, batch.targets)
            grad = loss.grad(predicted, batch.targets)
            net.backward(grad)
            optim.step(net)
        losses.append(epoch_loss)

        if verbose and (epoch % 100 == 0):
            print(epoch, epoch_loss)

        if val_data is not None and epoch % val_frequency == 0:
            val_pred = net.forward(val_data[0], training=False)
            val_loss = loss.loss(val_pred, val_data[1])
            val_losses.append(val_loss)

        if early_stopping:
            if epoch > early_stop_window and epoch % early_stop_frequency == 0:
                loss_window = losses[-early_stop_window:]
                if np.mean(loss_window[: early_stop_window // 10]) - np.mean(
                    loss_window[-early_stop_window // 10 :]
                ) < np.std(loss_window):
                    print(f"Early stopping after {epoch} epochs!")
                    return losses, val_losses
    return losses, val_losses

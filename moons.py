# coding=utf-8
from typing import List

import os

from sklearn.datasets import make_moons, make_blobs, make_circles
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

from magnn.loss import MSE
from magnn.nn import Net
from magnn.layer import Swish, Linear, Sigmoid, Tanh, Dropout, ReLu
from magnn.optimize import SGD
from magnn.train import train
from magnn.io import Scaler, BatchIterator

# cast envvar
VERBOSE=bool(os.getenv("VERBOSE", False))

Act = Swish


lr_init = 0.005
epochs_first = 1000
epochs_second = 100
val_frequency = 10
n_data = 5000
early_stopping = False
dropout_prob = 0.2

first_layer_width = 8
second_layer_width = 32
third_layer_width = 16

n_epochs_plot_1th = (epochs_first + epochs_second) // 10
batch_size = n_data // 10


losses = []


# CODE STARTS HERE
ds = make_moons(n_samples=n_data, noise=0.1)
# ds = make_circles(n_samples=n_data, noise=0.01, factor=0.3)


def encode_binary(y: int) -> List:
    if y == 1:
        return [0, 1]
    else:
        return [1, 0]


X, y = ds

y = np.array([encode_binary(_y) for _y in y])

X_t, X_test, y_t, y_test = train_test_split(X, y, test_size=0.15, shuffle=True)

X_train, X_val, y_train, y_val = train_test_split(
    X_t, y_t, test_size=0.10, shuffle=True
)

scaler = Scaler()
scaler.fit(X)

X_train = scaler.transform(X_train)

net = Net(
    [
        Linear(input_size=2, output_size=first_layer_width),
        Act(),
        # Dropout(prob=dropout_prob, size=first_layer_width),
        Linear(input_size=first_layer_width, output_size=second_layer_width),
        Act(),
        # Dropout(prob=dropout_prob, size=second_layer_width),
        # Linear(input_size=second_layer_width, output_size=third_layer_width),
        # Act(),
        # Dropout(prob=dropout_prob, size=third_layer_width),
        Linear(input_size=second_layer_width, output_size=2),
        Sigmoid(),
    ]
)

loss_1, loss_1_val = train(
    net,
    X_train,
    y_train,
    val_data=(X_val, y_val),
    val_frequency=val_frequency,
    epochs=epochs_first,
    iterator=BatchIterator(batch_size=100),
    optim=SGD(lr=lr_init),
    early_stopping=early_stopping,
    verbose=VERBOSE,
)

loss_2, loss_2_val = train(
    net,
    X_train,
    y_train,
    epochs=epochs_second,
    val_data=(X_val, y_val),
    val_frequency=val_frequency,
    iterator=BatchIterator(batch_size=100),
    optim=SGD(lr=lr_init * 0.1),
    early_stopping=early_stopping,
    verbose=VERBOSE,
)
# fix validation graph
loss_1_val = np.concatenate([[i] * val_frequency for i in loss_1_val])
loss_2_val = np.concatenate([[i] * val_frequency for i in loss_2_val])

if len(loss_1) != len(loss_1_val):
    loss_1_val = loss_1_val[: len(loss_1)]

if len(loss_2) != len(loss_2_val):
    loss_2_val = loss_2_val[: len(loss_2)]

losses = [loss_1, loss_2]
losses = np.concatenate(losses)
loss_val = np.concatenate([loss_1_val, loss_2_val])

test_pred = net.forward(X_test)
test_loss = MSE().loss(predicted=test_pred, actual=y_test)
correct = np.sum(
    np.argmax(test_pred, axis=1) == np.argmax(y_test, axis=1)
) / len(test_pred)

print(f"Training Loss:   {losses[-1]:.2f}")
print(f"Validation Loss: {loss_val[-1]:.2f}")
print(f"Test Loss:       {test_loss:.2f}")
print(f"Percent Correct: {correct:.2%}")

# correct = 0
# for x, y in zip(X_test, y_test):
#     preds = net.forward(x)
#     # print(np.round(preds, decimals=0), y)
#     # print(f"pred:\t{np.argmax(preds):.0f} -> {np.argmax(y):.0f}\t:true")
#     # print(np.round(preds, decimals=2), y)
#     if np.argmax(preds) == np.argmax(y):
#         correct += 1
#
# print(1.0 * correct / len(X_test))


fig, axes = plt.subplots(nrows=3, figsize=(6, 6))

axes[0].plot(losses, label="Loss")
axes[0].plot(loss_val, label="Val loss", alpha=0.5, c="r", ls="--")
axes[0].axvline(len(loss_1), c="k", alpha=0.5, label="LR changed")
axes[0].legend()

axes[1].plot(losses[-n_epochs_plot_1th:], label="Loss")
axes[1].legend()
axes[2].plot(loss_val[-n_epochs_plot_1th:], label="Val loss", ls="--")

axes[0].set_yscale("log")
axes[1].set_yscale("log")
axes[2].set_yscale("log")
fig.savefig("./circles_training.png")
fig.clf()


def visualize_decisions(X):
    from matplotlib.colors import ListedColormap

    h = 0.2
    fig, ax = plt.subplots()
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    cm_bright = ListedColormap(["#FF0000", "#0000FF"])

    ax.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=np.argmax(net.forward(X), axis=1),
        cmap=cm_bright,
        alpha=0.6,
        edgecolors="k",
    )

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())

    zz = net.forward(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    # zz = np.argmax(net.forward(np.c_[xx.ravel(), yy.ravel()]),axis=1)
    zz = zz.reshape(xx.shape)
    ax.contourf(xx, yy, zz, cmap=plt.cm.RdBu, alpha=0.3)


visualize_decisions(X_test,)


plt.savefig("./moons.png")

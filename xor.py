from magnn.optimize import SGD
from magnn.train import train
from magnn.nn import Net
from magnn.layer import Linear, Tanh, Swish, Sigmoid

import numpy as np


ins = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
targets = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])


net_1 = Net(
    [
        Linear(input_size=2, output_size=2),
        # Tanh(),
        Swish(),
        # Sigmoid(),
        Linear(input_size=2, output_size=2),
    ]
)


train(
    net_1, ins, targets, epochs=5000,
)  # optim=SGD(lr=0.00001))

for x, y in zip(ins, targets):
    preds = net_1.forward(x)
    print(x, preds, y)

from typing import List
import numpy as np
from magnn.train import train
from magnn.nn import Net
from magnn.layer import Linear, Tanh, Swish
from magnn.optimize import SGD

"""
Code to try to make FizzBuzz work by training an NN.
"""

def encode_fb(x: int) -> List:
    out = [0] * 4
    if x % 15 == 0:
        out[3] = 1
    elif x % 5 == 0:
        out[2] = 1
    elif x % 3 == 0:
        out[1] = 1
    else:
        out[0] = 1
    return out


def bin_enc(x: int) -> List:
    return [x >> i & 1 for i in range(10)]


inputs = np.array([bin_enc(x) for x in range(101, 1000)])

targets = np.array([encode_fb(x) for x in range(101, 1000)])


# Net time

net = Net(
    [
        Linear(input_size=10, output_size=20),
        Swish(),
        Linear(input_size=20, output_size=4),
    ]
)


train(net, inputs, targets, epochs=5000, optim=SGD(lr=0.00001))

for x in range(1, 50):
    pred = list(net.forward(bin_enc(x)))
    pred_idx = pred.index(max(pred))
    true = encode_fb(x)
    true_idx = true.index(max(true))
    labels = [str(x), "Fizz", "buzz", "fizzbuzz"]
    print(x, labels[pred_idx], labels[true_idx])

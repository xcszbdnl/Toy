from __future__ import print_function

import mxnet as mx
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon
from mxnet.gluon import nn
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import optimizer
mpl.rcParams['figure.dpi'] = 120


num_examples = 1000
num_inputs = 2
mx.random.seed(1)
random.seed(1)

true_w = [2, -3.4]
true_b = 4.2

X = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
y += 0.01 * nd.random.normal(shape=(num_examples))

dataset = gluon.data.ArrayDataset(X, y)


net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize()

square_loss = gluon.loss.L2Loss()


def train(batch_size, lr, epochs, period):
    assert period >= batch_size and period % batch_size == 0
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
    data_iter = gluon.data.DataLoader(dataset, batch_size, shuffle=True)
    total_loss = [np.mean(square_loss(net(X), y).asnumpy())]
    for epoch in range(1, epochs + 1):
        if epoch > 2:
            trainer.set_learning_rate(trainer.learning_rate * 0.1)
        for batch_i, (data, label) in enumerate(data_iter):
            with autograd.record():
                output = net(data)
                loss = square_loss(output, label)
            loss.backward()
            trainer.step(batch_size)
            if batch_i * batch_size % period == 0:
                total_loss.append(np.mean(square_loss(net(X), y).asnumpy()))
        print("Batch size %d, Learning rate %f, Epoch %d, loss %.4e" % (
            batch_size, lr, epoch, total_loss[-1]
        ))
    print("w:", np.reshape(net[0].weight.data().asnumpy(), (1, -1)),
          "b:", net[0].bias.data().asnumpy()[0])
    print("Total loss length:", len(total_loss))
    x_axis = np.linspace(0, epochs, len(total_loss), endpoint=True)
    plt.semilogy(x_axis, total_loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

train(batch_size=10, lr=0.1, epochs=3, period=10)



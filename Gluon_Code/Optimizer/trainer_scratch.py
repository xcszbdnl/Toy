from __future__ import print_function

import mxnet as mx
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon
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

def data_iter(batch_size):
    idx = list(range(num_examples))
    random.shuffle(idx)
    for batch_i, i in enumerate(range(0, num_examples, batch_size)):
        j = nd.array(idx[i: min(i + batch_size, num_examples)])
        yield batch_i, X.take(j), y.take(j)

def sgd_init_params():
    w = nd.random.normal(scale=1, shape=(num_inputs, 1))
    b = nd.zeros(shape=(1,))
    params = [w, b]
    for param in params:
        param.attach_grad()
    return params

def sgd_momentum_init_params():
    w = nd.random.normal(scale=1, shape=(num_inputs, 1))
    b = nd.zeros(shape=(1,))
    params = [w, b]
    vs = []
    for param in params:
        param.attach_grad()
        vs.append(param.zeros_like())
    return params, vs

def adagrad_init_params():
    w = nd.random.normal(scale=1, shape=(num_inputs, 1))
    b = nd.zeros(shape=(1,))
    params = [w, b]
    sqrs = []
    for param in params:
        param.attach_grad()
        sqrs.append(param.zeros_like())
    return params, sqrs

def adadelta_init_params():
    w = nd.random.normal(scale=1, shape=(num_inputs, 1))
    b = nd.zeros(shape=(1,))
    params = [w, b]
    sqrs = []
    deltas = []
    for param in params:
        param.attach_grad()
        sqrs.append(param.zeros_like())
        deltas.append(param.zeros_like())
    return params, sqrs, deltas

def adam_init_params():
    w = nd.random.normal(scale=1, shape=(num_inputs, 1))
    b = nd.zeros(shape=(1,))
    params = [w, b]
    sqrs = []
    vs = []
    for param in params:
        param.attach_grad()
        sqrs.append(param.zeros_like())
        vs.append(param.zeros_like())
    return params, sqrs, vs

def net(X, w, b):
    return nd.dot(X, w) + b


def square_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def train(batch_size, lr, epochs, period):
    assert period >= batch_size and period % batch_size == 0
    params, sqrs, vs = adam_init_params()
    w = params[0]
    b = params[1]
    total_loss = [np.mean(square_loss(net(X, w, b), y).asnumpy())]
    t = 0
    for epoch in range(1, epochs + 1):
        if epoch > 2:
            lr *= 0.1
        for batch_i, data, label in data_iter(batch_size):
            with autograd.record():
                output = net(data, w, b)
                loss = square_loss(output, label)
            loss.backward()
            t += 1
            optimizer.adam(params, sqrs, vs, batch_size, lr, t)
            if batch_i * batch_size % period == 0:
                total_loss.append(np.mean(square_loss(net(X,w ,b), y).asnumpy()))
        print("Batch size %d, Learning rate %f, Epoch %d, loss %.4e" % (
            batch_size, lr, epoch, total_loss[-1]
        ))
    print("w:", np.reshape(w.asnumpy(), (1, -1)),
          "b:", b.asnumpy()[0])
    print("Total loss length:", len(total_loss))
    x_axis = np.linspace(0, epochs, len(total_loss), endpoint=True)
    plt.semilogy(x_axis, total_loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

train(batch_size=10, lr=0.1, epochs=3, period=10)



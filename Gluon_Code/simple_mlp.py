from __future__ import print_function
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon
from utils import loadMnistData, sgd, accuracy, evaluate_accuracy


batch_size = 8

train_iter, test_iter = loadMnistData(batch_size)

num_hidden = 256
num_inputs = 28 * 28

num_outputs = 10

weight_scale = 0.1
w1 = nd.random_normal(shape=(num_inputs, num_hidden), scale=weight_scale)
b1 = nd.zeros(shape=(num_hidden))

w2 = nd.random_normal(shape=(num_hidden, num_outputs), scale=weight_scale)
b2 = nd.zeros(shape=(num_outputs))

params = [w1, b1, w2, b2]

for param in params:
    param.attach_grad()

def relu(X):
    return nd.maximum(X, 0)

def net(data):
    h1 = nd.dot(data.reshape((-1, num_inputs)), w1) + b1
    h1 = relu(h1)
    output = nd.dot(h1, w2) + b2
    return output

learing_rate = 0.1

softmax_cross_loss = gluon.loss.SoftmaxCrossEntropyLoss()

epochs = 5
for epoch in range(epochs):
    total_loss = .0
    total_acc = .0
    for data, label in train_iter:
        with autograd.record():
            output = net(data)
            loss = softmax_cross_loss(output, label)
        loss.backward()
        sgd(params, learing_rate / batch_size)

        total_loss += nd.mean(loss).asscalar()
        total_acc += accuracy(output, label)
    test_acc = evaluate_accuracy(test_iter, net)
    print('Epoch %d, Train Loss: %f, Train Acc: %f, Test Acc: %f'% (
        epoch, total_loss / len(train_iter), total_acc / len(train_iter), test_acc))
from __future__ import print_function

from mxnet import nd
from mxnet import autograd
from mxnet import gluon
import utils

# w = nd.arange(4).reshape((1, 1, 2, 2))
# b = nd.array([1])
#
# data = nd.arange(9).reshape((1, 1, 3, 3))
#
# out = nd.Convolution(data, w, b, kernel=w.shape[2:], num_filter=w.shape[1])
#
# print(data, '\n\n', w, '\n\n', b, '\n\n', out )

weight_scale = .01

ctx = utils.getCtx()

w1 = nd.random_normal(shape=(20, 1, 5, 5), ctx=ctx, scale=weight_scale)

b1 = nd.zeros(w1.shape[0], ctx=ctx)

w2 = nd.random_normal(shape=(50, 20, 3, 3), scale=weight_scale, ctx=ctx)

b2 = nd.zeros(w2.shape[0], ctx=ctx)

w3 = nd.random_normal(shape=(1250, 128), scale=weight_scale, ctx=ctx)

b3 = nd.zeros(w3.shape[1], ctx=ctx)

w4 = nd.random_normal(shape=(128, 10), scale=weight_scale, ctx=ctx)

b4 = nd.zeros(shape=w4.shape[1], ctx=ctx)

params = [w1, b1, w2, b2, w3, b3, w4, b4]

for param in params:
    param.attach_grad()

def net(X, verbose=False):
    X = X.as_in_context(w1.context)
    h1_conv = nd.Convolution(data=X, weight=w1, bias=b1, kernel=w1.shape[2:], num_filter=w1.shape[0])
    h1_activation = nd.relu(h1_conv)
    h1 = nd.Pooling(data=h1_activation, pool_type='max', kernel=(2,2), stride=(2,2))

    h2_conv = nd.Convolution(data=h1, weight=w2, bias=b2, kernel=w2.shape[2:], num_filter=w2.shape[0])
    h2_activation = nd.relu(h2_conv)
    h2 = nd.Pooling(data=h2_activation, pool_type='max', kernel=(2,2), stride=(2,2))
    h2 = nd.flatten(h2)

    h3_linear = nd.dot(h2, w3) + b3
    h3 = nd.relu(h3_linear)

    h4_linear = nd.dot(h3, w4) + b4
    if verbose is True:
        print('1st conv shape', h1.shape)
        print('2nd conv shape', h2.shape)
        print('1st dense shape', h3.shape)
        print('2nd dense shape', h4_linear.shape)
        print('output:', h4_linear)
    return h4_linear

batch_size = 256
train_iter, test_iter = utils.loadMnistData(batch_size)

for data, label in train_iter:
    net(data, verbose=True)
    break

softmax_loss = gluon.loss.SoftmaxCrossEntropyLoss()
learning_rate = 0.5

epochs = 5

for epoch in range(epochs):
    total_loss = .0
    total_acc = .0
    for data, label in train_iter:
        with autograd.record():
            output = net(data)
            loss = softmax_loss(output, label)
        loss.backward()
        utils.sgd(params, learning_rate / batch_size)

        total_loss += nd.mean(loss).asscalar()
        total_acc += utils.accuracy(output, label)
    test_acc = utils.evaluate_accuracy(test_iter, net)
    print('Epoch %d, Train loss: %f, Train acc: %f, Test acc: %f\n' % (
        epoch, total_loss / len(train_iter), total_acc / len(train_iter), test_acc
    ))
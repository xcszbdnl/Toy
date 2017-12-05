from __future__ import print_function

from mxnet import gluon
from mxnet import autograd
from mxnet import ndarray as nd
import utils

z = nd.arange(16).reshape((2, 2, 2, 2))
print(z[:, 0, :, :])
print(z[:, 1, :, :])

y = z.mean(axis=(0, 2), keepdims=True)
print(y)

def pure_batch_norm(X, gamma, beta, eps=1e-5):
    assert len(X.shape) in (2, 4)
    if len(X.shape) == 2:
        mean = X.mean(axis=0)
        variance = ((X - mean) ** 2).mean(axis=0)
    else:
        mean = X.mean(axis=(0, 2, 3), keepdims=True)
        variance = ((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
    X_hat = (X - mean) / nd.sqrt(variance + eps)
    return gamma.reshape(mean.shape) * X_hat + beta.reshape(mean.shape)

A = nd.arange(6).reshape((3, 2))

print(A)
print(pure_batch_norm(A, gamma=nd.array([1, 1]), beta=nd.array([0, 0])))

def batch_normal(X, gamma, beta, is_training, moving_mean, moving_variance,
                 eps=1e-5, moving_momentum=0.9):
    assert len(X.shape) in (2, 4)
    if len(X.shape) == 2:
        mean = X.mean(axis=0)
        variance = ((X - mean) ** 2).mean(axis=0)
    else:
        mean = X.mean(axis=(0, 2, 3), keepdims=True)
        variance = ((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
        moving_mean = moving_mean.reshape(mean.shape)
        moving_variance = moving_variance.reshape(mean.shape)
    if is_training:
        X_hat = (X - mean) / (eps + variance)
        moving_mean[:] = moving_mean * moving_momentum + (
            1 - moving_momentum) * mean
        moving_variance[:] = moving_variance * moving_momentum + (
            1 - moving_momentum) * variance
    else:
        X_hat = (X - moving_mean) / (variance + moving_momentum)
    return gamma.reshape(mean.shape) * X_hat + beta.reshape(mean.shape)

weight_scale = 0.1
ctx = utils.getCtx()
c1 = 20
gamma1 = nd.random.normal(shape=(c1), scale=weight_scale, ctx=ctx)
beta1 = nd.random.normal(shape=(c1), scale=weight_scale, ctx=ctx)
moving_mean1 = nd.zeros(shape=gamma1.shape, ctx=ctx)
moving_variance1 = nd.zeros(shape=gamma1.shape, ctx=ctx)

w1 = nd.random.normal(shape=(c1, 1, 5, 5), scale=weight_scale, ctx=ctx)
b1 = nd.zeros(shape=(c1), ctx=ctx)

c2 = 50
gamma2 = nd.random.normal(shape=(c2), scale=weight_scale, ctx=ctx)
beta2 = nd.random.normal(shape=(c2), scale=weight_scale, ctx=ctx)
moving_mean2 = nd.zeros(shape=gamma2.shape, ctx=ctx)
moving_variance2 = nd.zeros(shape=gamma2.shape, ctx=ctx)

w2 = nd.random.normal(shape=(c2, c1, 3, 3), scale=weight_scale, ctx=ctx)
b2 = nd.zeros(shape=c2, ctx=ctx)

o3 = 128
# gamma3 = nd.random.normal(shape=o3, scale=weight_scale, ctx=ctx)
# beta3 = nd.random.normal(shape=o3, scale=weight_scale, ctx=ctx)
# moving_mean3 = nd.zeros(shape=o3, ctx=ctx)
# moving_variance3 = nd.zeros(shape=o3, ctx=ctx)

w3 = nd.random.normal(shape=(1250, o3), scale=weight_scale, ctx=ctx)
b3 = nd.zeros(shape=o3, ctx=ctx)

w4 = nd.random.normal(shape=(o3, 10), scale=weight_scale, ctx=ctx)
b4 = nd.zeros(shape=10, ctx=ctx)

params = [w1, b1, w2, b2, w3, b3, w4, b4]

for param in params:
    param.attach_grad()

def net(X, is_training=False, verbose=False):
    X = X.as_in_context(w1.context)

    h1_conv = nd.Convolution(
        data=X, weight=w1, bias=b1, kernel=w1.shape[2:], num_filter=c1)
    h1_bn = batch_normal(h1_conv, gamma1, beta1, is_training, moving_mean1, moving_variance1)
    h1_activation = nd.relu(h1_bn)
    h1 = nd.Pooling(data=h1_activation, pool_type='max', kernel=(2, 2), stride=(2, 2))
    h2_conv = nd.Convolution(
        data=h1, weight=w2, bias=b2, kernel=w2.shape[2:], num_filter=c2)
    h2_bn = batch_normal(h2_conv, gamma2, beta2, is_training, moving_mean2, moving_variance2)
    h2_activation = nd.relu(h2_bn)
    h2 = nd.Pooling(data=h2_activation, pool_type='max', kernel=(2, 2), stride=(2, 2))
    h2 = nd.Flatten(h2)
    h3_dense = nd.dot(h2, w3) + b3
    h3_activation = nd.relu(h3_dense)

    output = nd.dot(h3_activation, w4) + b4
    if verbose:
        print('1st conv block:', h1_conv.shape)
        print('2st conv block:', )
    return output

softmax_loss = gluon.loss.SoftmaxCrossEntropyLoss()
learning_rate = 0.5
batch_size = 128
train_iter, test_iter = utils.loadMnistData()

epochs = 5

for epoch in range(epochs):
    total_loss = .0
    total_acc = .0
    for data, label in train_iter:
        with autograd.record():
            output = net(data, is_training=True)
            loss = softmax_loss(output, label)
        loss.backward()
        utils.sgd(params, learning_rate / batch_size)

        total_loss += nd.mean(loss).asscalar()
        total_acc += utils.accuracy(output, label)
    test_acc = utils.evaluate_accuracy(test_iter, net)
    print('Epoch %d, Train loss: %f, Train acc: %f, Test acc: %f' % (
        epoch, total_loss / len(train_iter), total_acc / len(train_iter), test_acc
    ))


from __future__ import print_function
import mxnet as mx
from mxnet import ndarray as nd
from mxnet import gluon
from mxnet import image

def accuracy(output, label):
    return nd.mean(output.argmax(axis=1) == label).asscalar()

def evaluate_accuracy(data_iter, net):
    acc = 0
    for data, label in data_iter:
        output = net(data)
        acc += accuracy(output, label)
    return acc / len(data_iter)

def sgd(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

def getCtx():
    try:
        ctx = mx.gpu()
        _ = nd.zeros((2), ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx


def loadMnistData(batch_size=16, resize=None):
    def transform(data, label):
        if resize:
            data = image.imresize(data, resize, resize)
        return data.astype('float32').transpose((2, 0, 1)) / 255.0, label.astype('float32')
    train_data = gluon.data.vision.FashionMNIST(train=True, transform=transform)
    test_data = gluon.data.vision.FashionMNIST(train=False, transform=transform)

    train_iter = gluon.data.DataLoader(train_data, batch_size, shuffle=True)
    test_iter = gluon.data.DataLoader(test_data, batch_size, shuffle=False)

    return train_iter, test_iter


def load_data_fashion_mnist(batch_size, resize=None, root="~/.mxnet/datasets/fashion-mnist"):
    """download the fashion mnist dataest and then load into memory"""
    def transform_mnist(data, label):
        # transform a batch of examples
        if resize:
            n = data.shape[0]
            new_data = nd.zeros((n, resize, resize, data.shape[3]))
            for i in range(n):
                new_data[i] = image.imresize(data[i], resize, resize)
            data = new_data
        # change data from batch x height x weight x channel to batch x channel x height x weight
        return nd.transpose(data.astype('float32'), (0,3,1,2))/255, label.astype('float32')
    mnist_train = gluon.data.vision.FashionMNIST(root=root, train=True, transform=transform_mnist)
    mnist_test = gluon.data.vision.FashionMNIST(root=root, train=False, transform=transform_mnist)
    train_data = gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
    test_data = gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)
    return (train_data, test_data)
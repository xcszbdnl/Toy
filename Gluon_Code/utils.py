from mxnet import ndarray as nd
from mxnet import gluon

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

def transform(data, label):
    return data.astype('float32') / 255.0, label.astype('float32')

def loadMnistData(batch_size=16):
    train_data = gluon.data.vision.FashionMNIST(train=True, transform=transform)
    test_data = gluon.data.vision.FashionMNIST(train=False, transform=transform)

    train_iter = gluon.data.DataLoader(train_data, batch_size, shuffle=True)
    test_iter = gluon.data.DataLoader(test_data, batch_size, shuffle=False)

    return train_iter, test_iter
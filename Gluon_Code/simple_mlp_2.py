from __future__ import print_function
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon
from utils import accuracy, evaluate_accuracy, sgd
import matplotlib.pyplot as plt

def transform(data, label):
    return data.astype('float32') / 255, label.astype('float32')

mnist_train = gluon.data.vision.FashionMNIST(train=True, transform=transform)
mnist_test = gluon.data.vision.FashionMNIST(train=False, transform=transform)


data, label = mnist_train[0]
print('example shape:', data.shape, 'label:', label)

def show_images(images):
    n = images.shape[0]
    _, figs = plt.subplots(1, n, figsize=(15, 15))
    for i in range(n):
        figs[i].imshow(images[i, :].reshape((28, 28)).asnumpy())
        figs[i].axes.get_xaxis().set_visible(False)
        figs[i].axes.get_yaxis().set_visible(False)
    plt.show()

def get_text_labels(label):
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress,', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
    ]
    return [text_labels[int(i)] for i in label]

data, label = mnist_train[0:9]
# show_images(data)
# print(get_text_labels(label))


batch_size = 10

train_iter = gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
test_iter = gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)

num_inputs = 28 * 28
num_outputs = 10

W = nd.random_normal(shape=(num_inputs, num_outputs))
b = nd.random_normal(shape=(num_outputs,))

params = [W, b]

for param in params:
    param.attach_grad()

def softmax(X):
    exp = nd.exp(X)
    partition = exp.sum(axis=1, keepdims=1)
    return exp / partition

# x = nd.random_normal(shape=(3, 5))
# print(softmax(x).sum(axis=1))

def cross_entropy(yhat, y):
    return - nd.pick(nd.log(yhat), y)

def net(X):
    return softmax(nd.dot(X.reshape((-1, num_inputs)), W) + b)

epochs = 10
base_rate = 0.001

for epoch in range(epochs):
    train_loss = .0
    train_acc = .0
    for data, label in train_iter:
        with autograd.record():
            output = net(data)
            loss = cross_entropy(output, label)
        loss.backward()
        learning_rate = base_rate / (epoch + 1)
        sgd(params, learning_rate)
        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(output, label)
    test_acc = evaluate_accuracy(test_iter, net)
    print('Epoch %d. Loss: %f, Train acc: %f, Test acc:%f' %(
        epoch, train_loss / len(train_iter), train_acc / len(train_iter), test_acc
    ))








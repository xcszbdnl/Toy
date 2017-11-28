from __future__ import print_function
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon
from utils import accuracy, evaluate_accuracy
def transform(data, label):
    return data.astype('float32') / 255.0, label.astype('float32')

mnist_train = gluon.data.vision.FashionMNIST(train=True, transform=transform)
mnist_test = gluon.data.vision.FashionMNIST(train=False, transform=transform)

batch_size = 256

train_iter = gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
test_iter = gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)

for data, label in train_iter:
    print(data.shape, label.asnumpy())
    break

net = gluon.nn.Sequential()

with net.name_scope():
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(10))

net.initialize()

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

epochs = 5

for epoch in range(epochs):
    train_loss = .0
    train_acc = .0
    for data, label in train_iter:
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(output, label)
    test_acc = evaluate_accuracy(test_iter, net)
    print('Epoch %d, Train loss: %f, Train acc: %f, Test acc: %f' % (
        epoch, train_loss / len(train_iter), train_acc / len(train_iter), test_acc))


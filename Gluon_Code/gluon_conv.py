from __future__ import print_function
from mxnet.gluon import nn, loss, Trainer
from mxnet import ndarray as nd
from mxnet import autograd
import utils


def get_net():
    net = nn.Sequential()
    with net.name_scope():
        net.add(
            nn.Conv2D(channels=20, kernel_size=5, activation='relu'),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Conv2D(channels=50, kernel_size=3, activation='relu'),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Flatten(),
            nn.Dense(128, activation='relu'),
            nn.Dense(10)
        )
    return net

net = get_net()
net.initialize()
batch_size = 128
train_iter, test_iter = utils.loadMnistData(batch_size)

for data, label in train_iter:
    print(data)
    break

softmax_loss = loss.SoftmaxCrossEntropyLoss()

trainer = Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})

epochs = 5

for epoch in range(epochs):
    total_loss = .0
    total_acc = .0
    for data, label in train_iter:
        with autograd.record():
            output = net(data)
            losses = softmax_loss(output, label)
        losses.backward()
        trainer.step(batch_size)

        total_loss += nd.mean(losses).asscalar()
        total_acc += utils.accuracy(output, label)

    test_acc = utils.evaluate_accuracy(test_iter, net)
    print('Epoch %d, Train Loss: %f, Train acc: %f, Test acc: %f\n' % (
        epoch, total_loss / len(train_iter), total_acc / len(train_iter), test_acc
    ))



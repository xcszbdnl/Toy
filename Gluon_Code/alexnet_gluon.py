from __future__ import print_function
from mxnet import ndarray as nd
from mxnet import gluon
from mxnet import init
from mxnet import autograd
from mxnet.gluon import nn, loss, Trainer
import utils

def get_net():
    net = nn.Sequential()
    with net.name_scope():
        net.add(
            # first stage
            nn.Conv2D(channels=96, kernel_size=11,
                      strides=4, activation='relu'),
            nn.MaxPool2D(pool_size=3, strides=2),
            # second stage
            nn.Conv2D(channels=256, kernel_size=5,
                      padding=2, activation='relu'),
            nn.MaxPool2D(pool_size=3, strides=2),
            # third stage
            nn.Conv2D(channels=384, kernel_size=3,
                      padding=1, activation='relu'),
            nn.Conv2D(channels=384, kernel_size=3,
                      padding=1, activation='relu'),
            nn.Conv2D(channels=384, kernel_size=3,
                      padding=1, activation='relu'),
            nn.MaxPool2D(pool_size=3, strides=2),
            nn.Flatten(),
            nn.Dense(4096, activation='relu'),
            nn.Dropout(.5),
            nn.Dense(4096, activation='relu'),
            nn.Dropout(.5),
            nn.Dense(10)
        )
    return net

batch_size = 64
train_iter, test_iter = utils.loadMnistData(batch_size, resize=224)
# train_iter, test_iter = utils.load_data_fashion_mnist(batch_size, resize=224)

# for data, label in train_iter:
#     print(data.shape)
#     break
net = get_net()
ctx = utils.getCtx()

net.initialize(ctx=ctx, init=init.Xavier())

softmax_loss = loss.SoftmaxCrossEntropyLoss()

epochs = 5

trainer = Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})

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
    print('Epoch %d, Train loss: %f, Train acc: %f, Test acc: %f' % (
        epoch, total_loss / len(train_iter), total_acc / len(train_iter), test_acc
    ))

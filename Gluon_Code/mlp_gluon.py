from __future__ import print_function
from mxnet import ndarray as nd
from mxnet import gluon
from mxnet import autograd
from utils import accuracy, evaluate_accuracy, loadMnistData

batch_size = 256
num_hidden = 256

train_iter, test_iter = loadMnistData(batch_size)

net = gluon.nn.Sequential()

with net.name_scope():
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(num_hidden, activation='relu'))
    net.add(gluon.nn.Dense(10))

net.initialize()

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})

softmax_loss = gluon.loss.SoftmaxCrossEntropyLoss()

epochs = 5
for epoch in range(epochs):
    total_loss = .0
    total_acc = .0
    for data, label in train_iter:
        with autograd.record():
            output = net(data)
            loss = softmax_loss(output, label)
        output.backward()
        trainer.step(batch_size)

        total_loss += nd.mean(loss).asscalar()
        total_acc += accuracy(output, label)
    test_acc = evaluate_accuracy(test_iter, net)
    print('Epoch %d, Train Loss: %f, Train acc: %f, Test acc:%f' % (
        epoch, total_loss / len(train_iter), total_acc / len(train_iter), test_acc
    ))

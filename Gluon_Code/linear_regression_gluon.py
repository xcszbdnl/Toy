from __future__ import print_function
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon

num_inputs = 2
num_examples = 1000

true_w = [2, -3.4]
true_b = 4.2

X = nd.random_normal(shape=(num_examples, num_inputs))
y = nd.dot(X, nd.array(true_w)) + true_b

y += 0.01 * nd.random_normal(shape=y.shape)

batch_size = 10
dataset = gluon.data.ArrayDataset(X, y)

data_iter = gluon.data.DataLoader(dataset, batch_size, shuffle=True)

for data, label in data_iter:
    print(data, label)
    break

net = gluon.nn.Sequential()
net.add(gluon.nn.Dense(1))

net.initialize()

square_loss =gluon.loss.L2Loss()

trainer = gluon.Trainer(
    net.collect_params(), 'sgd', {'learning_rate': 0.1}
)

epochs = 5

for epoch in range(epochs):
    total_loss = 0
    for data, label in data_iter:
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        trainer.step(batch_size)
        total_loss += nd.sum(loss).asscalar()
    print('Epoch %d, average loss:%.f' % (epoch, total_loss / num_examples))

print(true_w, net[0].weight.data())
print(true_b, net[0].bias.data())

help(trainer.step)
help(net[0].weight)
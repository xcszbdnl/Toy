from __future__ import print_function
from mxnet import autograd
from mxnet import ndarray as nd
from mxnet.gluon import nn

net = nn.Sequential()

with net.name_scope():
    net.add(nn.Dense(256, activation='relu'))
    net.add(nn.Dense(10))

print(net)

class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        with self.name_scope():
            self.dense1 = nn.Dense(10)
            self.dense0 = nn.Dense(256)

    def forward(self, x):
        return self.dense1(nd.relu(self.dense0(x)))

net2 = MLP()
print(net2)
net2.initialize()

x = nd.random.normal(shape=(4, 20))

print(net2(x))

class Sequential(nn.Block):
    def __init__(self, **kwargs):
        super(Sequential, self).__init__(**kwargs)
    def add(self, block):
        self._children.append(block)

    def forward(self, x):
        for block in self._children:
            x = block(x)
        return x

net4 = Sequential()

net4.add(nn.Dense(256))
net4.add(nn.Dense(10))
net4.initialize()
y = net4(x)

print('net4:', y)

class RecMLP(nn.Block):
    def __init__(self, **kwargs):
        super(RecMLP, self).__init__(**kwargs)
        self.net = nn.Sequential()
        with self.name_scope():
            self.denses = [nn.Dense(256), nn.Dense(128), nn.Dense(64)]

    def forward(self, x):
        for dense in self.denses:
            x = dense(x)
        return x

rec_mlp = RecMLP()
y = rec_mlp(x)
print(y)

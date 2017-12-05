from __future__ import print_function
from mxnet.gluon import nn
from mxnet import ndarray as nd


def get_net():
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Dense(10))
        net.add(nn.Dense(100))
    return net

x = nd.random.normal(shape=(3, 5))

net = get_net()
params = net.collect_params()
net.params
print(params)
net.initialize()

y = net(x)
print(y)
w = net[0].weight
b = net[0].bias
print('weight:', w.data())
print('bias:', b.data())
print('w grad:', w.grad())
print('bias grad:', b.grad())

from mxnet import init

params = net.collect_params()
print(params)

params.initialize(init=init.Normal(sigma=0.2), force_reinit=True)
print(net[0].weight.data(), net[0].bias.data())

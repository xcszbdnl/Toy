from __future__ import print_function
from mxnet.gluon import nn
from mxnet import init
from mxnet import ndarray as nd

def get_net():
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Dense(4))
    return net

class MyInit1(init.Initializer):
    def __init__(self):
        super(MyInit1, self).__init__()

    def _init_weight(self, _, arr):
        print('init weight', arr.shape)
        nd.random.uniform(5, 10, out=arr)

x = nd.random.normal(shape=(5, 5))
net = get_net()
net.initialize(MyInit1())

net(x)
print(net[0].weight.data())



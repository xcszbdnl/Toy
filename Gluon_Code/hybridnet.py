from __future__ import print_function
from mxnet.gluon import nn
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd

class HybridNet(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(HybridNet, self).__init__(**kwargs)
        with self.name_scope():
            self.fc1 = nn.Dense(10)
            self.fc2 = nn.Dense(2)

    def hybrid_forward(self, F, x):
        print(F)
        print(x)
        x = F.relu(self.fc1(x))
        print(x)
        return self.fc2(x)


net = HybridNet()
net.initialize()

x = nd.random.normal(shape=(1, 4))

y = net(x)
print(y)

net.hybridize()
y = net(x)
print("-------------")
y = net(x)
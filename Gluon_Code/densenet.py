from __future__ import print_function
from mxnet.gluon import nn
from mxnet import gluon
from mxnet import nd


def conv_block(channels):
    out = nn.Sequential()
    out.add(
        nn.BatchNorm(),
        nn.Activation('relu'),
        nn.Conv2D(channels, kernel_size=3, padding=1)
    )
    return out


class DenseBlock(nn.Block):
    def __init__(self, growth_rate, layers, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        self.net = nn.Sequential()
        for i in range(layers):
            self.net.add(conv_block(growth_rate))

    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = nd.concat(x, out, dim=1)
        return x

def transition_block(channels):
    out = nn.Sequential()
    out.add(
        nn.BatchNorm(),
        nn.Activation('relu'),
        nn.Conv2D(channels, kernel_size=1),
        nn.AvgPool2D(pool_size=2, strides=2)
    )
    return out

# blk = DenseBlock(2, 10)
# blk.initialize()

x = nd.random.uniform(shape=(4, 3, 5, 5))

# print(blk(x).shape)

# tblk = transition_block(10)
# tblk.initialize()
# print(tblk(x).shape)

init_channels = 64
growth_rate = 32
block_layers = [6, 12, 24, 16]


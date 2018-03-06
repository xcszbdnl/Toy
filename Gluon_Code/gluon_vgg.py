from __future__ import print_function
from mxnet import gluon
from mxnet import ndarray as nd
def vgg_block(num_convs, channels):
    out = gluon.nn.Sequential()
    for _ in range(num_convs):
        out.add(
            gluon.nn.Conv2D(channels=channels, kernel_size=3,
                            padding=1, activation='relu')
        )
        out.add(
            gluon.nn.MaxPool2D(pool_size=2, strides=2)
        )
    return out

blk = vgg_block(2, 128)

blk.initialize()

x = nd.random.normal(shape=(2, 3, 32, 32))

y = blk(x)

print(y.shape)

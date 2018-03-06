from mxnet import gluon
from mxnet.gluon import nn
from mxnet import ndarray as nd


def mlpconv(channels, kernel_size, pooling, padding, strides=1, max_pooling=True):
    out = nn.Sequential()
    out.add(
        nn.Conv2D(channels=channels, kernel_size=kernel_size, strides=strides, padding=padding,
                  activation='relu')
        
    )


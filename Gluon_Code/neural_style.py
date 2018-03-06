from __future__ import print_function
from mxnet import gluon
from mxnet import ndarray as nd

x = nd.random.normal(shape=(1, 3, 4, 4))
c = x.shape[1]



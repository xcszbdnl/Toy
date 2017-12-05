from __future__ import print_function
from mxnet import gluon
from mxnet import ndarray as nd

class MyDense(gluon.nn.Block):
    def __init__(self, units, in_units, **kwargs):
        super(MyDense, self).__init__(**kwargs)
        with self.name_scope():
            self.weight = self.params.get(
                'weight', shape=(in_units, units))
            self.bias = self.params.get('bias', shape=(units,))

    def forward(self, x):
        linear = nd.dot(x, self.weight.data()) + self.bias.data()
        return linear

dense = MyDense(5, in_units=10, prefix='my_dense')

dense.initialize()
print(dense.params)

x = nd.random.normal(shape=(5, 10))

y = dense(x)

print(y)

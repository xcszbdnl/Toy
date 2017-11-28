import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
z = np.ones((3, 4))
mx_z = nd.array(z)
z_np = mx_z.asnumpy()
print mx_z
print z_np
x = nd.ones((3, 4))
y = nd.random_normal(0, 1, shape=(3,4))
print x + y
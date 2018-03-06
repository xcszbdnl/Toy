from __future__ import print_function
from mxnet import ndarray as nd
from time import time
from mxnet.gluon import nn
from mxnet import gluon
import os
import subprocess
# start = time()
# x = nd.random.normal(shape=(1000, 1000))
# y = nd.dot(x, x)
#
# print("-----work spend time:%f------" % (time() - start))
#
# print(y)
#
# print("-----work spend time:%f------" % (time() - start))

def data():
    start = time()
    batch_size = 1024
    for _ in range(60):
        if _ % 10 == 0:
            print("Batch %d, time %f sec", time() - start)
        x = nd.ones(shape=(batch_size, 1024))
        y = nd.ones(shape=(batch_size))
        yield x, y

net = nn.Sequential()
with net.name_scope():
    net.add(
        nn.Dense(1024, activation='relu'),
        nn.Dense(1024, activation='relu'),
        nn.Dense(1)
    )

net.initialize()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {})

loss = gluon.loss.L2Loss()

def get_mem():
    res = subprocess.check_output(['ps', 'u', '-p', str(os.getpid())])
    return int(str(res).split()[15]) / 1e3

for x, y in data():
    break

loss(y, net(x)).wait_to_read()

mem = get_mem()

for x, y in data():
    loss(y, net(x)).wait_to_read()

print("Increased memory: %f MB" % (get_mem() - mem))


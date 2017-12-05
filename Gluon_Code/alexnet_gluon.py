from mxnet import ndarray as nd
from mxnet import gluon
from mxnet.gluon import nn, loss, Trainer
import utils

def get_net():
    net = nn.Sequential()
    with net.name_scope():
        net.add(
            # first stage
            nn.Conv2D(channels=96, kernel_size=11,
                      strides=4, activation='relu'),
            nn.MaxPool2D(pool_size=3, strides=2),
            # second stage
            nn.Conv2D(channels=256, kernel_size=5,
                      padding=2, activation='relu'),
            nn.MaxPool2D(pool_size=3, strides=2),
            # third stage
            nn.Conv2D(channels=384, kernel_size=3,
                      padding=1, activation='relu'),
            nn.Conv2D(channels=384, kernel_size=3,
                      padding=1, activation='relu'),
            nn.Conv2D(channels=384, kernel_size=3,
                      padding=1, activation='relu'),
            nn.MaxPool2D(pool_size=3, strides=2),
            nn.Flatten(),
            nn.Dense(4096, activation='relu'),
            nn.Dropout(.5),
            nn.Dense(4096, activation='relu'),
            nn.Dropout(.5),
            nn.Dense(10)
        )
    return net

batch_size = 64
# train_iter, test_iter = utils.loadMnistData(batch_size, resize=224)
train_iter, test_iter = utils.load_data_fashion_mnist(batch_size, resize=224)

for data, label in train_iter:
    print(data.shape)
    break

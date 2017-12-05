from __future__ import print_function
import pandas as pd
from mxnet import ndarray as nd
from mxnet import gluon
from mxnet import autograd
import matplotlib.pyplot as plt
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

all_X = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'],
                   test.loc[:, 'MSSubClass': 'SaleCondition']))

print(train.head(), train.shape, test.shape)

numeric_feats = all_X.dtypes[all_X.dtypes != 'object'].index

all_X[numeric_feats] = all_X[numeric_feats].apply(lambda x: (x - x.mean) / (x.std()))

all_X = pd.get_dummies(all_X, dummy_na=True)

all_X = all_X.fillna(all_X.mean())

num_train = train.shape[0]

X_train = all_X[:num_train].asmatrix()
X_test = all_X[num_train:].asmatrix()

y_train = train.SalePrice.asmatrix()

square_loss = gluon.loss.L2Loss()

def get_rmse_log(net, X_train, y_train):
    num_train = X_train.shape[0]
    clipped_preds = nd.clip(net(X_train), 1, float('inf'))
    return np.sqrt(2 * nd.sum(square_loss(
        nd.log(clipped_preds), nd.log(y_train))).asscalar() / num_train)

def get_net():
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(1))
    net.initialize()
    return net

def train(net, X_train, y_train, X_test, y_test, epochs, verbose_epoch, learning_rate, weight_decay):
    train_loss = []
    if X_test is not None:
        test_loss = []
    batch_size = 100
    dataset_train = gluon.data.ArrayDataset(X_train, y_train)
    train_iter = gluon.data.DataLoader(dataset_train, batch_size, shuffle=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': learning_rate, 'wd': weight_decay})


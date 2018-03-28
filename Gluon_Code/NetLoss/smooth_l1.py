from mxnet import gluon
from mxnet import ndarray as nd
import numpy as np
import matplotlib
matplotlib.rcParams['figure.dpi'] = 120
import matplotlib.pyplot as plt


class SmoothL1Loss(gluon.loss.Loss):
    def __init__(self, scale=1.0, batch_axis=0, **kwargs):
        super(SmoothL1Loss, self).__init__(None, batch_axis, **kwargs)
        self._scale = scale

    def hybrid_forward(self, F, output, label, mask):
        loss = F.smooth_l1((output - label) * mask, scalar=self._scale)
        return loss.mean(axis=self._batch_axis, exclude=True)


if __name__ == '__main__':
    colors = ['blue', 'red', 'green', 'black']
    scales = [.5, 1, 10]

    x = nd.arange(-2, 2, .01)
    for i, s in enumerate(scales):
        y = nd.smooth_l1(x, scalar=s)
        plt.plot(x.asnumpy(), y.asnumpy(), colors[i])

    y = x ** 2
    plt.plot(x.asnumpy(), y.asnumpy(), 'black')

    plt.legend(['scale='+str(s) for s in scales] + ['Square loss'])
    plt.show()


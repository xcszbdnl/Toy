from mxnet import gluon
from mxnet import ndarray as nd
import numpy as np
import matplotlib.pyplot as plt

colors = ['blue', 'green', 'red', 'black', 'magenta']


def focal_loss(gamma, x):
    return -(1 - x)**gamma * np.log(x)


class FocalLoss(gluon.loss.Loss):
    def __init__(self, axis=-1, alpha=0.25, gamma=2, batch_axis=0, **kwargs):
        super(FocalLoss, self).__init__(None, batch_axis, **kwargs)
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma

    def hybrid_forward(self, F, output, label):
        output = F.softmax(output)
        pj = output.pick(label, axis=self._axis, keepdims=True)
        loss = -self._alpha * (1 - pj) ** gamma * pj.log()
        return loss.mean(axis=self._batch_axis, exclude=True)

x = nd.array([
    [0, 1],
    [2, 3],
    [4, 5]
])

y = x.mean(axis=0, exclude=False)
print(y)


x = np.arange(0.01, 1.0, .01)
gammas = [0, 0.25, 0.5, 1.0]

for i, gamma in enumerate(gammas):
    plt.plot(x, focal_loss(gamma, x), colors[i])

plt.legend(['gamma=' + str(gamma) for gamma in gammas])
# plt.show()


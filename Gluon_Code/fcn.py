import os
import tarfile
from mxnet import gluon
from mxnet import image
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision
import numpy as np
import utils
from mxnet import nd
data_root = '../../Dataset'
voc_root = data_root + '/VOC2007'
url = ('http://host.robots.ox.ac.uk/pascal/VOC/voc2012'
       '/VOCtrainval_11-May-2012.tar')
sha1 = '4e443f8a2eca6b1dac8a6c57641b67dd40621a49'

# fname = gluon.utils.download(url, data_root, sha1_hash=sha1)
#
# if not os.path.isfile(voc_root + '/ImageSets/Segmentation/train.txt'):
#     with tarfile.open(fname, 'r') as f:
#         f.extractall(data_root)


def read_images(root=voc_root, train=True):
    txt_name = root + '/ImageSets/Segmentation/' + (
        'train.txt' if train else 'val.txt'
    )

    with open(txt_name) as f:
        images = f.read().split()

    n = len(images)
    data, label = [None] * n, [None] * n
    for i, fname in enumerate(images):
        data[i] = image.imread('%s/JPEGImages/%s.jpg' % (
            root, fname
        ))
        label[i] = image.imread('%s/SegmentationClass/%s.png' % (
            root, fname
        ))
    return data, label

train_images, train_labels = read_images()

imgs = []
for i in range(3):
    imgs += [train_images[i], train_labels[i]]

# utils.show_images(imgs, nrows=3, ncols=2)

def rand_crop(data, label, height, width):
    data, rect = image.random_crop(data, (width, height))
    label = image.fixed_crop(label, *rect)
    return data, label

imgs = []
for i in range(3):
    imgs += rand_crop(train_images[i], train_labels[i], 200, 300)

# utils.show_images(imgs, nrows=3, ncols=2)

classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'potted plant',
           'sheep', 'sofa', 'train', 'tv/monitor']
# RGB color for each class
colormap = [[0,0,0],[128,0,0],[0,128,0], [128,128,0], [0,0,128],
            [128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],
            [64,128,0],[192,128,0],[64,0,128],[192,0,128],
            [64,128,128],[192,128,128],[0,64,0],[128,64,0],
            [0,192,0],[128,192,0],[0,64,128]]

cm2lbl = np.zeros(256 ** 3)
for i, cm in enumerate(colormap):
    cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i

def img2label(im):
    data = im.astype('int32').asnumpy()
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    return nd.array(cm2lbl[idx])

y = img2label(train_labels[0])

# print(y[105:115, 130:140])

rgb_mean = nd.array([0.485, 0.456, 0.406])
rgb_std = nd.array([0.229, 0.224, 0.225])

def normalize_image(im):
    return (im.astype('float32') / 255.0 - rgb_mean) / rgb_std

class VocSegDataset(gluon.data.Dataset):

    def _filter(self, images):
        return [im for im in images if (
            im.shape[0] >= self.crop_size[0] and
            im.shape[1] >= self.crop_size[1]
        )]

    def __init__(self, train, crop_size):
        # super(VocSegDataset, self).__init__()
        self.crop_size = crop_size
        data, label = read_images(train=train)
        data = self._filter(data)
        self.data = [normalize_image(im) for im in data]
        self.label = self._filter(label)
        print('Read ' + str(len(self.data)) + ' examples')

    def __getitem__(self, idx):
        data, label = rand_crop(self.data[idx],
                                self.label[idx],
                                *self.crop_size)
        data = data.transpose((2, 0, 1))
        label = img2label(label)
        return data, label

    def __len__(self):
        return len(self.data)

input_shape = (320, 480)
voc_train = VocSegDataset(True, input_shape)
voc_test = VocSegDataset(False, input_shape)

batch_size = 64
train_data = gluon.data.DataLoader(voc_train, batch_size=batch_size,
                                   shuffle=True, last_batch='discard')
test_data = gluon.data.DataLoader(voc_test, batch_size=batch_size,
                                  shuffle=False, last_batch='discard')

conv = nn.Conv2D(10, kernel_size=4, strides=2, padding=1)
conv_trans = nn.Conv2DTranspose(3, kernel_size=4, strides=2, padding=1)

conv.initialize()
conv_trans.initialize()

x = nd.random.uniform(shape=(1, 3, 64, 64))
y = conv(x)
print('Input:', x.shape)
print('After conv: ', y.shape)
print('After conv tranpose: ', conv_trans(y).shape)

pretrained_net = vision.get_model('resnet18_v1', pretrained=True)
print(pretrained_net.features[-4:])
net = nn.HybridSequential()

for layer in pretrained_net.features[-2:]:
    net.add(layer)

x = nd.random.uniform(shape=(1, 3, *input_shape))
print('Input shape: ', x.shape)
# y = net(x)
#
# print(y.shape)


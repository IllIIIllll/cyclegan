from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.misc

def _to_range(images, min_value=0.0, max_value=1.0, dtype=None):
    assert \
        np.min(images) >= -1.0 - 1e-5 and np.max(images) <= 1.0 + 1e-5 \
        and (images.dtype == np.float32 or images.dtype == np.float64), \
        'The input images should be float64(32) and in the range of [-1.0, 1.0]!'
    if dtype is None:
        dtype = images.dtype
    return ((images + 1.) / 2. * (max_value - min_value) + min_value).astype(dtype)

def _im2uint(images):
    return _to_range(images, 0, 255, np.uint8)

def imread(path, mode='RGB'):
    return scipy.misc.imread(path, mode=mode) / 127.5 - 1

def imwrite(image, path):
    return scipy.misc.imsave(path, _to_range(image, 0, 255, np.uint8))

def imresize(image, size, interp='bilinear'):
    return (scipy.misc.imresize(_im2uint(image), size, interp=interp) / 127.5 - 1).astype(image.dtype)

def immerge(images, row, col):
    if images.ndim == 4:
        c = images.shape[3]
    elif images.ndim == 3:
        c = 1

    h, w = images.shape[1], images.shape[2]
    if c > 1:
        img = np.zeros((h * row, w * col, c))
    else:
        img = np.zeros((h * row, w * col))
    for idx, image in enumerate(images):
        i = idx % col
        j = idx // col
        img[j * h:j * h + h, i * w:i * w + w, ...] = image

    return img

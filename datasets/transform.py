import pdb

from datasets import transform as base_transform
import numpy as np
import random

from util_tools import is_list, is_dict, get_valid_args


class NoOperation():
    def __call__(self, x):
        return x


class BaseSilTransform():
    def __init__(self, disvor=255.0, img_shape=None):
        self.disvor = disvor
        self.img_shape = img_shape

    def __call__(self, x):
        if self.img_shape is not None:
            s = x.shape[0]
            _ = [s] + [*self.img_shape]
            x = x.reshape(*_)
        return x / self.disvor


class BaseSilCuttingTransform():
    def __init__(self, img_w=64, disvor=255.0, cutting=None):
        self.img_w = img_w
        self.disvor = disvor
        self.cutting = cutting

    def __call__(self, x):
        if self.cutting is not None:
            cutting = self.cutting
        else:
            cutting = int(self.img_w // 64) * 10
        x = x[..., cutting:-cutting]
        return x / self.disvor

class RandomCropTransform():
    def __init__(self, img_w=64, padding=4, disvor=255.0, cutting=None):
        self.img_w = img_w
        self.disvor = disvor
        self.cutting = cutting
        self.padding = padding

    def __call__(self, x):
        if self.cutting is not None:
            cutting = self.cutting
        else:
            cutting = int(self.img_w // 64) * 10
        # padding
        c, h, w = x.shape
        out_h, out_w = h, w - 2 * cutting
        x = np.pad(x, ((0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant', constant_values=0)
        #random crop
        c, h, w = x.shape
        i = random.randint(0, h - out_h)
        j = random.randint(0, w - out_w)

        x = x[..., i: i + out_h, j: j + out_w]
        return x / self.disvor

class BaseRgbTransform():
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = [0.485*255, 0.456*255, 0.406*255]
        if std is None:
            std = [0.229*255, 0.224*255, 0.225*255]
        self.mean = np.array(mean).reshape((1, 3, 1, 1))
        self.std = np.array(std).reshape((1, 3, 1, 1))

    def __call__(self, x):
        return (x - self.mean) / self.std

